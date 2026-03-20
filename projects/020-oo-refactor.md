# 项目 P020（修订版 v1.2）：架构重构（面向对象 + 可测试 + 渐进迁移）

状态：Phase 0→Phase 5 已完成（2026‑03‑21）  
作者：Cline（AI 软件工程师）  
修订：v1.2 合入评审建议（Pydantic 校验 / 依赖注入 / Façade API 合约 / JAX‑First 设计）  
关联文档与代码：
- docs/12-code-architecture-and-apis.md（现有架构与模块职责）
- scripts/run_simulation.py（现行主循环）
- pygcm/*（核心物理模块）
- projects/016-jax-acceleration.md（JAX 兼容与加速路径）
- 所有 docs/04（运行时参数目录与环境变量），docs/06/07/08/09/10/11/14/15/16/18（物理与生态设计）

本文件目标：将“面向对象重构”的愿景落为可执行蓝图，明确对象边界、API 合约、迁移阶段、测试与验收标准、兼容策略与性能预算，确保以最小风险渐进完成重构。v1.2 在 v1.1 的基础上，落实了“以代码强制的边界约束”“依赖注入提升可测性”“Façade 即 API 合约”“JAX‑First 纯函数化”四项增强。


## 0. 设计原则（与现有仓库一致，v1.2 强化）

- 渐进迁移：禁止“大爆炸”式重写。新增 OO 层首先作为 façade 包裹现有实现，逐步内收。
- 单一职责：对象封装“状态 + 行为”；物理公式与数值算子尽量做成纯函数模块（stateless）。
- 可测试：每个模块都有可独立构造的最小状态与可重复调用的 API；提供契约/回归测试。
- 向后兼容：环境变量、CLI 用法、输出产物默认不变；新 API 通过适配层注入。
- 守恒优先：任何阶段的修改都不得破坏 docs/06/08/09 规定的能量/水量闭合标准。
- 性能预算：重构后端到端步时开销不增加（±5% 范围内）；新增层次不可引入不必要复制。

v1.2 追加：
- JAX‑First：核心数值/物理算子按“纯函数 + 无副作用 + 定形数组”组织，优先满足 `@jit` 编译、`vmap` 向量化的要求（详见 §10、§2“纯函数模块”）。
- DI（依赖注入）优先：`QingdaiWorld` 通过构造函数接收子系统实例，测试可注入 mock/stub；提供 `create_default()` 工厂方法封装默认装配（详见 §2、§4）。
- 状态“伪不可变”策略：推荐 `step()` 返回新的 `WorldState`（值语义），主循环赋值替换，以杜绝隐式共享副作用；性能敏感段可局部采用 in‑place 写入（详见 §1）。
- Façade = API 合约：临时 façade 类的公开方法签名应与目标类完全一致，保障“热切换无代码改动”（详见 §3.5）。


## 1. 概念边界（Configuration / Parameter / State）与“以代码强制”的实现

与 docs/12 对齐，但在工程层面明确“生命周期与存储”与“代码层约束”：

- Configuration（配置，运行前确定，运行期不可变）
  - 例：网格分辨率（n_lat, n_lon）、dt、数值方案/开关（QD_FILTER_TYPE, QD_USE_OCEAN）。
  - 表达：不可变模型 `SimConfig`。
  - v1.2：建议采用 **Pydantic** `BaseModel`（或 v2 `BaseModel`/`pydantic-settings`）代替纯 dataclass，用于加载时的**运行时类型检查与值验证**（如 `dt_seconds > 0`、`filter_type ∈ {"combo","hyper4","shapiro","spectral"}` 等）。
    ```python
    # 示例（文档内伪代码）
    from pydantic import BaseModel, field_validator
    class SimConfig(BaseModel, frozen=True):
        n_lat: int
        n_lon: int
        dt_seconds: float
        filter_type: str
        use_ocean: bool = True
        @field_validator("dt_seconds")
        @classmethod
        def _dt_pos(cls, v): 
            assert v > 0, "dt_seconds must be > 0"
            return v
        @field_validator("filter_type")
        @classmethod
        def _ft_ok(cls, v):
            assert v in {"combo","hyper4","shapiro","spectral"}
            return v
    ```
- Parameter（参数，定义规律，单次实验通常恒定，可由实验更改）
  - 例：物理常数、温室参数、Bowen 比、生态基因库、光谱带定义。
  - 表达：不可变/可版本化模型 `PhysicsParams`, `EcologyParams`, `SpectralBands`（同样推荐采用 Pydantic，用于枚举/范围验证）。
- State（状态，时间快照）
  - 例：u, v, h, Ts, Ta, q, cloud, SST, uo, vo, η, h_ice, W_land, SWE, routing buffers, LAI/生态状态等。
  - 表达：可变容器 `WorldState` 拆分子状态；支持全量/分组持久化；含 schema_version。
  - v1.2 “伪不可变”策略：**值语义演化**。`QingdaiWorld.step()` 的推荐接口为：
    ```python
    def step(self) -> "WorldState":
        new_state = self._compute_next_state(self.state)
        self.state = new_state
        return new_state
    ```
    这样杜绝了时间步内的跨子系统隐式副作用；对性能敏感段，允许内部采用 in‑place 写数组，但必须保证对外暴露的 `WorldState` 替换为新的逻辑对象（copy‑on‑write 或 builder 模式）。该策略简化调试与回放，便于引入“状态快照对比”的测试工具。


## 1A. 状态管理与演化：双缓冲（Double Buffering）

为在保证“时间步不可变/原子更新”的同时兼顾性能，建议在 QingdaiWorld 内采用“双缓冲”机制管理 WorldState：

- 初始化：在启动时创建两个 WorldState 实例，分别记为 current_state 与 next_state。
- 单步演化（第 n 步）：
  - 所有子系统读取 current_state（视为只读），并将结果写入 next_state（构建“下一状态”）。
  - 子系统方法推荐形态：`subsystem.step(read_state, write_state, ...)`，或在对象内部通过 world.current_state/world.next_state 访问。
- 缓冲交换（Buffer Swap）：
  - 步末仅交换引用而非复制数据：`current_state, next_state = next_state, current_state`。
  - 通过该交换，避免了每步全量分配/复制，显著降低内存与 GC 压力。

此策略是“伪不可变”的工程化实现：在同一时间步内严格分离“读缓冲（current）”与“写缓冲（next）”，杜绝跨模块交错写入导致的脏读；同时复用内存，满足高性能需求。

推荐最小实现示意：
```python
class QingdaiWorld:
    def __init__(...):
        self.current_state = self._alloc_initial_state()
        self.next_state = self._alloc_initial_state()

    def step(self):
        # 读取 self.current_state，写入 self.next_state
        self.forcing.update(self.current_state, self.next_state)
        self.atmos.time_step(self.current_state, self.next_state)
        if self.config.use_ocean:
            self.ocean.step(self.current_state, self.next_state)
        self.hydro.step(self.current_state, self.next_state)
        if self.config.use_routing:
            self.route.step(self.current_state, self.next_state)
        if self.config.use_ecology:
            self.eco.step_subdaily(self.current_state, self.next_state)
            if self._hits_day_boundary():
                self.eco.step_daily(self.current_state, self.next_state)
        # 步末：时间推进+缓冲交换
        self.next_state.t_seconds = self.current_state.t_seconds + self.config.dt_seconds
        self.current_state, self.next_state = self.next_state, self.current_state
        return self.current_state
```

与“返回新状态”的纯函数式方案相比，双缓冲可避免每步大规模数组重新分配，更适合包含海量场变量的 GCM；在接口层面仍保持值语义（对外暴露“新状态”），并可与 JAX‑First 的纯函数内核配合（内核函数对输入/输出数组显式区分，满足 @jit 的副作用约束）。

## 2. 目标对象模型（类与模块）与 DI/JAX‑First

顶层对象（面板）：
- QingdaiWorld（有状态对象）
  - config: SimConfig（Pydantic）
  - params: ParamsRegistry（聚合 PhysicsParams/EcologyParams/SpectralBands）
  - grid: Grid
  - subsystems: Atmosphere, Ocean, Surface, Hydrology, Routing, Ecology, Forcing（见下）
  - state: WorldState（聚合各子状态）
  - DI（依赖注入）：构造函数接收“可选子系统实例”；未传入时由 `create_default()` 工厂创建默认实例（便于测试注入 mock）。
    ```python
    class QingdaiWorld:
        def __init__(self, config, params, grid,
                     atmos=None, ocean=None, surface=None, hydrology=None, routing=None, ecology=None, forcing=None,
                     state=None):
            self.config, self.params, self.grid = config, params, grid
            self.atmos = atmos or Atmosphere.default(config, params, grid)
            self.ocean = ocean or Ocean.default(config, params, grid)
            # ...
            self.state = state or self._alloc_initial_state()

        @classmethod
        def create_default(cls, config=None, params=None, grid=None) -> "QingdaiWorld":
            # 统一默认装配逻辑（可读 env，构造 grid/params 子对象）
            ...
    ```
  - methods:
    - step() -> WorldState：推进一个物理步（值语义，返回新状态）
    - run(n_steps|duration)
    - save_state(path) / load_state(path)
    - diagnostics() / plotters()

子系统（有状态对象）：
- Atmosphere / Ocean / Surface / Hydrology / Routing / Ecology / Forcing
  - 均提供 `.default(config, params, grid)` 以支撑工厂式装配；构造函数保持最小依赖并可被测试替换。
  - **JAX‑First**：数值核与物理公式置于纯函数模块（见下），对象方法完成组织/写回，便于 `@jit` 生效与单元测试可控。

纯函数模块（stateless，JAX 友好）：
- physics（docs/06 组合器）：shortwave, longwave, boundary_layer_fluxes, calculate_dynamic_albedo 等
- numerics（docs/10 算子）：laplacian_sphere, hyperdiffusion, shapiro, spectral_filters
- humidity（docs/08 核心算子）：q_sat, evaporation_flux, condensation
- hydrology_core（P009/P019 算子）：sigmoid phase partition, snow/melt, runoff
- ocean_core（P011 数值核）
- ecology_core（docs/13/14/15/16 带/吸收/聚合）  
- jax_compat（与 projects/016 一致）：xp, map_coordinates 替换件

（其余对象职责与 v1.1 一致，略）


## 3. 目录与文件布局（不破坏现有，新增 façade 层）

新增（建议）：
```
pygcm/
  world/
    __init__.py
    world.py           # QingdaiWorld（主编排）
    config.py          # SimConfig (Pydantic) + env/文件加载器
    params.py          # ParamsRegistry + 各 Params (Pydantic)
    state.py           # WorldState + 子状态定义、schema_version
    forcing_facade.py  # Forcing façade（组合 orbital/physics/spectral）
    adapters.py        # 旧脚本/模块的适配层（渐进迁移）
```

保留并渐进重构：
```
pygcm/
  dynamics.py          # 逐步内收进 Atmosphere（可先做轻薄代理）
  ocean.py             # 逐步内收进 Ocean
  hydrology.py         # 逐步内收进 Hydrology
  routing.py           # 保持，Routing 类化
  energy.py, physics.py, humidity.py, topography.py, forcing.py, numerics（新增）
  ecology/             # 维持结构；对接 Ecology 子系统
  jax_compat.py        # 继续保留
scripts/run_simulation.py  # 外观不变，逐步调用 world.QingdaiWorld
```

### 3.5 Façade = API 合约（v1.2 新增强制约束）

- 临时 façade（如 `AtmosphereFacade`）的**公开方法签名**（方法名、参数列表、返回类型）**必须与最终 `Atmosphere` 类完全一致**。
- 目的：确保后续从 façade 切换到真实实现时，`QingdaiWorld` 零改动；façade 不仅是临时代理，更是“提前冻结的 API 合约”。
- 审核机制：对 façade 与目标类的签名进行单元测试级反射比对（如 `inspect.signature`）。


## 4. API 合约（签名草案，含 DI 与工厂）

关键模型（以 Pydantic 表达；文档示意）
```python
from pydantic import BaseModel, field_validator
class SimConfig(BaseModel, frozen=True): ...
class PhysicsParams(BaseModel, frozen=True): ...
class SpectralBands(BaseModel, frozen=True): ...
class EcologyParams(BaseModel, frozen=True): ...
class ParamsRegistry(BaseModel, frozen=True):
    physics: PhysicsParams
    bands: SpectralBands
    ecology: EcologyParams
```

状态容器（值语义，伪不可变策略）
```python
@dataclass
class AtmosState: u: np.ndarray; v: np.ndarray; h: np.ndarray; Ta: np.ndarray; q: np.ndarray; cloud: np.ndarray
# ... 其它子状态同 v1.1
@dataclass
class WorldState: ...
```

世界对象（DI + 工厂 + 值语义）
```python
class QingdaiWorld:
    def __init__(self, config: SimConfig, params: ParamsRegistry, grid: Grid,
                 atmos: Optional[Atmosphere]=None, ocean: Optional[Ocean]=None, ...,
                 state: Optional[WorldState]=None): ...
    @classmethod
    def create_default(cls, config: Optional[SimConfig]=None, params: Optional[ParamsRegistry]=None, grid: Optional[Grid]=None) -> "QingdaiWorld": ...
    def step(self) -> WorldState: ...
    def run(self, n_steps: Optional[int]=None, duration_days: Optional[float]=None) -> None: ...
```

时序（统一顺序，兼容 docs/12；同 v1.1，略）


## 5. 现有模块映射（从 → 到）

| 现有 | 新结构中的归属 | 策略 |
|---|---|---|
| dynamics.py | Atmosphere + numerics | 先 façade 代理，再逐步内收 time_step/反噪到子类 |
| energy.py/physics.py | physics（纯函数模块） + Surface | 直接复用函数；Surface 仅封装状态与组合 |
| humidity.py | humidity（纯函数） + Atmosphere 集成 | 复用；在 Atmosphere 中组织调用与写回 |
| ocean.py | Ocean + numerics | 先 façade，保持现有接口，迁移 step/风应力/极点修正 |
| hydrology.py | Hydrology + hydrology_core | 将相态/雪/桶拆纯函数，Hydrology 负责读写状态 |
| routing.py | Routing（类化已基本具备） | 统一接口，缓存/诊断对齐 |
| ecology/* | Ecology（保持结构） | Adapter 对接 world；逐步替换直连 run_simulation |
| forcing.py/orbital.py | Forcing façade + physics.solar | 组合输出 I / T_eq / 带化 I_b（docs/14） |
| topography.py | Surface 初始化/属性图生成 | 初始化加载/插值逻辑不变 |


## 6. 向后兼容策略

- 运行入口不变：`python3 -m scripts.run_simulation`；环境变量目录（docs/04）保持有效。
- 第一阶段：`scripts/run_simulation.py` 仅新增“if USE_WORLD: world = QingdaiWorld(...); world.run()”，默认仍走旧路径；以环境变量门控迁移（例如 QD_USE_OO=1）。
- 全部图像/输出目录/命名保持；diagnostics 文本格式保留关键行（用于既有分析脚本）。
- Restart 文件：保持旧字段，同时新增 group 化结构（见 §8）；使用 schema_version 做迁移。


## 7. 迁移阶段（Milestones）

- 阶段 0（façade 注入，完成）
  - 新增 `pygcm/world/*`（config/params/state/world）骨架；
  - `QingdaiWorld` 内部直接调用旧脚本逻辑（adapter 直连 scripts.run_simulation 的函数块或现有类）；
  - 新增 `QD_USE_OO=1` 开关，默认关闭；冒烟测试通过。

- 阶段 1（最小世界编排可运行，完成）
  - 接入 `QingdaiWorld.run()` 最小编排，使用 `world.state` 与 `Atmosphere` orchestrator 驱动一步；
  - 构建网格与基础地形、摩擦与热容量图；热力强迫计算 `Teq` 并推进；
  - 保持 `QD_USE_OO` 门控，`QD_USE_OO_STRICT=1` 时可严格走 OO 路径；
  - 已接入 `Hydrology`/`Ecology` 到 OO 世界步进：降水相态分配、SWE/桶模型更新、生态子步反照率反馈、日步 soil-index 更新；
  - 已补齐 Coupler 端口最小集（insolation/u10/v10）并在 world 编排中传递。

- 阶段 2（Forcing/Physics 纯函数化，完成）
  - 将短波/长波/BL/Teq/反照率组合器从过程调用改为纯函数模块（保留原逻辑，JAX‑First 编写）；
  - Atmosphere/Surface/Ocean 改为仅组织调用 + 写入 state。
  - 已完成首批配置固化：`SimConfig` 增加正值校验（网格与步长）与 `snapshot()`，`create_default()` 输出配置快照用于运行追踪。
  - 已补齐参数注册首批固化：`ParamsRegistry.from_env()` 与 `PhysicsParams/SpectralBands/EcologyParams` 校验及快照；运行可选输出 metadata JSON（`QD_OO_METADATA_JSON`）。
  - 已开始参数驱动替换 legacy 读取：OO `run()` 优先使用 `self.params.physics/ecology` 驱动 MLD、热容、湿度初值、BL 参数、Bowen 比与诊断频率，减少直接 `os.getenv` 读取。
  - 本批继续推进到湿度/水文注入层：OO 路径直接构造 `HumidityParams/HydrologyParams` 并注入，`snowpack_step` 改为使用参数对象中的 `snow_albedo_fresh`，进一步降低子模块内部 env 耦合。
  - 本批继续推进到 Coupler 闭环：`Coupler` 支持 `energy_params/humidity_params` 显式注入并提供 `set_external_params()`；默认参数改为类默认值（不再经 env getter），OO world 侧统一注入参数对象。
  - 本批继续推进到运行控制闭环：`world.run()` 对总时长与 metadata 开关不再直接读取 env，统一由 `SimConfig` 字段（`total_years/sim_days/oo_metadata_json/oo_config_diag`）驱动。
  - 本批继续推进运行控制收口：`SimConfig` 新增 `oo_metadata_enable/default_orbit_fraction`，metadata 写出与默认仿真时长完全由配置对象控制。
  - 本批继续收紧配置边界：`SimConfig.snapshot` 改为 `domain/runtime_control` 分组 schema，`ParamsRegistry.snapshot` 改为 `parameter_groups` 分组 schema，并补齐 roundtrip 序列化一致性测试。

- 阶段 3（Atmosphere/Ocean 子系统内收，4–7 天）
  - 将 dynamics/ocean 的 time_step/step 逻辑迁入类方法；numerics 保持独立；
  - 保持 façcade 兼容；步时与闭合逐步回归。
  - 已启动首批：新增 `world/ocean_orchestrator.py`，将海洋推进与状态写回从 `world.run()` 主循环中内收为可注入 orchestrator。
  - 本批继续推进：移除 `world.run()` 中 ocean fallback 分支，统一收敛到 orchestrator 接口（`configure + step_and_write`），并提供 legacy backend 自动包装兼容。
  - 本批继续推进 Atmosphere 同级收敛：新增 `world/atmos_orchestrator.py`，`world.run()` 大气路径统一为 orchestrator 接口（`configure + step_and_write`），并提供 legacy `time_step` 自动包装兼容。
  - 本批继续推进 Phase 3 收口：将 Atmos/Ocean `configure` 参数统一收敛为单一 `WorldOrchestratorConfigureSpec` 对象，`world.run()` 不再分散拼装两套参数。
  - 本批完成 Phase 3 最后一段收口：新增 spec builder（`build_world_orchestrator_spec`），将 `orchestrator_spec` 组装逻辑下沉并进一步瘦身 `world.run()`。

- 阶段 4（Hydrology/Routing/Ecology 对齐，5–8 天）
  - P009/P019 算子拆分；Routing 收束为类 API；Ecology 用 Adapter 对接 world 时序；
  - 日界/子步的调度固化到 world。
  - 已启动首批：新增 `RoutingOrchestrator` 并接入 `world.run()`，支持 injected/legacy routing 统一适配。
  - 本批继续推进：新增 `HydrologyOrchestrator` 并接入 `world.run()`，固定执行顺序为 `hydrology → routing → ecology`。
  - 本批继续推进：新增 `EcologyOrchestrator` 并内收生态调度，形成统一三段编排链路（`hydrology → routing → ecology`）。
  - 本批继续推进：聚合 routing/hydrology/ecology diagnostics 为统一 `world_diagnostics` 输出结构，替代分散步内打印并降低后处理耦合。
  - 本批继续推进：`world_diagnostics` 增加可选 JSON 落盘与 schema 版本化校验，降低后处理耦合并稳定跨版本消费。
  - 本批继续推进：将 `world_diagnostics` schema 抽象为独立 typed contract（dataclass），并增加严格字段校验与可选向后兼容策略。
  - Phase 4 验收清单（收口）：
    - [x] Routing/Hydrology/Ecology 全部纳入 orchestrator 链路，支持 injected/legacy 适配
    - [x] 固定顺序编排稳定：`hydrology → routing → ecology`（含顺序回归）
    - [x] `world.run()` 去散落式子系统调用，形成统一编排入口
    - [x] 聚合诊断统一为 `world_diagnostics`，覆盖 energy/water + hydrology/routing/ecology
    - [x] `world_diagnostics` 支持 schema 校验、typed contract、可选 JSON 落盘与 backward-compat 解析
    - [x] 分批回归 + M4 回归持续通过（稳定指标未退化）

- 阶段 5（JAX 互操作 + 性能与基准，5–7 天）
  - 与 projects/016 的 `xp` 后端对齐；核心算子加 `@jit`；
  - 端到端对比基准、内存占用与步时日志。
  - 本批完成：补齐 JAX 运行策略开关（含 CPU/Metal 可选启用），并保持无 JAX 环境下稳定回退。
  - 本批完成：增强 `scripts/benchmark_jax.py`，输出内存峰值与结构化 JSON 基准结果。
  - 本批完成：新增 Phase5 回归测试（benchmark JSON 输出与 JAX 兼容层开关导入稳定性）。
  - Phase 5 验收清单（收口）：
    - [x] JAX/Numpy 路径可通过开关稳定切换，且默认保留安全回退
    - [x] 基准脚本输出总时长/步时/内存峰值，并支持 JSON 产物
    - [x] CI 本地回归覆盖新增 Phase5 核心契约测试
    - [x] P016/P020/README 状态与运行说明同步

每阶段均包含“止损点”：任何异常可回滚开关（QD_USE_OO=0）；保持产线可运行。


## 8. 状态持久化与 Schema

- 容器：NetCDF（主）、NPZ（轻量 autosave）；按 group 划分：/atmos, /ocean, /surface, /hydro, /routing, /ecology, /meta
- 元数据（/meta）：
  - schema_version（int）、git_hash、created_at_utc、grid dims、config snapshot、params snapshot
  - topo source/land fraction/albedo/friction stats 对齐 README/log
- 迁移工具：io_netcdf.migrate(path, from_version→to_version)，容忍少字段/新增字段默认化
- 安全写：tmp + fsync + atomic replace；滚动备份 N 份（按 docs/15 autosave 经验）  
- 与旧格式兼容：load() 优先尝试新 schema，否则读取旧字段 + 构造缺省 group；打印黄色兼容日志。


## 9. 测试矩阵与验收标准（v1.2 补充 DI/Mock 场景）

- 单元（pytest/numba-free）：
  - numerics：laplacian, hyperdiffuse, shapiro（谱/空间一致性）
  - physics：SW/LW/BL 等通量维度与典型输入输出范围
  - hydrology_core：相态 Sigmoid、SWE/melt 守恒
  - ecology_core：带积分能量/反射与聚合限幅
  - io：schema 读写/迁移/原子写容错
- 契约（contract tests）：
  - façade vs 目标类签名一致性测试（反射比对）
  - 子系统 API：Atmosphere/Ocean/Hydrology/Routing/Ecology 的输入/输出字段与副作用范围
  - Forcing：给定 time/params，输出 I/I_b 的确定性（对随机种子固定）
- 回归（integration/regression）：
  - 基线片段对比：能量闭合、潜热一致、水量闭合；图像 Golden（SSIM）
- 性能：
  - 步时统计：OO 开关前后 Δt_step 在 ±5% 内；内存峰值不高于 +10%
  - JAX 路径（可选）：CPU 单步下降 ≥ 30%（目标），GPU/TPU 更佳
- DI/Mock：
  - 利用依赖注入，将 `QingdaiWorld` 注入“伪造 Atmosphere/Ocean”等子系统，隔离测试单一模块的行为与边界条件响应。


## 10. 性能与内存策略（JAX‑First 强调）

- 禁止重复分配：状态数组在 WorldState 内创建，子系统拿视图/引用；纯函数返回写入目标数组（out 参数）。
- IO 分批：大数组按需序列化；诊断分辨率/频率可降采样。
- **JAX‑First**：核心算子以纯函数方式实现，禁用隐式全局状态；可选 `xp=jax.numpy` 后端；绘图/NetCDF 前显式 `np.asarray`。
- 计算图粒度：将小函数合入大核（如 combined radiation）降低 Python 调度开销。
- 值语义 + in‑place 的平衡：对外暴露值语义；内部在需要时安全地 in‑place 更新，再在返回前构造新 `WorldState`（builder/copy‑on‑write）。


## 11. 最小代码骨架（可立即落地，含 DI）

```python
# pygcm/world/world.py
class QingdaiWorld:
    def __init__(self, config, params, grid,
                 atmos=None, ocean=None, surface=None, hydrology=None, routing=None, ecology=None, forcing=None,
                 state=None):
        self.config, self.params, self.grid = config, params, grid
        self.atmos = atmos or AtmosphereFacade(self)   # façade 与目标类签名一致
        self.ocean = ocean or OceanFacade(self)
        self.hydro = hydrology or HydrologyFacade(self)
        self.route = routing or RoutingFacade(self)
        self.eco = ecology or EcologyFacade(self)
        self.forcing = forcing or ForcingFacade(self)
        self.state = state or self._alloc_initial_state()

    @classmethod
    def create_default(cls, config=None, params=None, grid=None):
        # 解析 env → SimConfig/ParamsRegistry（Pydantic 校验）
        # 构造 Grid；构建默认子系统实例；分配初始状态
        ...

    def step(self):
        # 以 builder 模式构造下一状态（值语义）
        self.forcing.update()
        self.atmos.time_step()
        if self.config.use_ocean:
            self.ocean.step()
        self.hydro.step()
        if self.config.use_routing:
            self.route.step()
        if self.config.use_ecology:
            self.eco.step_subdaily()
            if self._hits_day_boundary(): self.eco.step_daily()
        self._maybe_plot()
        self.state.t_seconds += self.config.dt_seconds
        return self.state
```

> 说明：Facade 初期直接调用旧模块函数，保证接入 0 风险；随后逐步替换为新类逻辑。公开签名与目标类完全一致以履行“API 合约”。


## 12. 风险与对策

- 风险：接口漂移导致产线中断  
  对策：QD_USE_OO 守门；façade 与目标类签名一致；阶段化回归测试。
- 风险：守恒退化  
  对策：每阶段引入“守恒回归”并将失败视为阻断；只在通过后推进。
- 风险：性能下降  
  对策：性能基准纳入每阶段退出标准；profiling 针对热点做回退或 JAX 化；内外值语义平衡。
- 风险：持久化不兼容  
  对策：引入 schema_version 与迁移工具；保持旧格式读取路径与默认填充；黄色兼容日志。


## 13. 时间表（建议）

- Week 1：阶段 0–1（façade + 配置/参数/状态 Pydantic 校验）；冒烟回归
- Week 2：阶段 2（Forcing/Physics 纯函数化，JAX‑First）；基线回归
- Week 3：阶段 3（Atmosphere/Ocean 内收）；性能回归
- Week 4：阶段 4（Hydrology/Routing/Ecology 对齐）；端到端年尺度试跑
- Week 5：阶段 5（JAX 互操作 + 文档/示例/README 更新）

> 所有阶段可弹性并行（Ocean 与 Ecology 可并行），但合入主干前必须通过守恒与回归。


## 14. 交付与文档更新

- 本设计文档（P020）纳入仓库；每阶段结束补齐“变更记录与默认组变化”；
- 更新 docs/12 与 README 中“开发者指南/运行 GCM”一节，标注 OO 开关与迁移状态；
- 新增开发者示例：如何用 `QingdaiWorld` 以 10 行代码跑起一次短程仿真；如何在 test harness 中构建最小 WorldState 做单元/集成测试。


## 15. 变更记录（Changelog）

- 2025‑09‑27：v1.2 合入评审建议：Pydantic 校验、DI 依赖注入、Façade 即 API 合约、JAX‑First 强化；补充不可变状态策略与 Mock 测试用例建议。
- 2025‑09‑27：v1.1 可落地蓝图：对象模型、API 合约、迁移阶段、测试矩阵、持久化 schema、性能策略与骨架代码；对齐 docs/12/04/06/07/08/09/10/11/14/15/16/18。
- 2025‑11‑22：v1.3 启动 Phase 1 最小世界编排：`QingdaiWorld.run()` 使用 `world.state` + `Atmosphere`（legacy_spectral 引擎）推进，读取 env 构建地形、热容量与 `Teq`，在 `QD_USE_OO` 开启时可运行最小 OO 路径。
- 2025‑11‑22：v1.4 扩展 Phase 1：OO 世界接入 Hydrology/Ecology 调度（SWE/桶更新与生态反照率反馈）、Coupler 端口扩展（insolation/u10/v10）并完成严格模式短程回归。
- 2026‑03‑20：v1.5 推进 Phase 2 首批：`SimConfig` 新增校验与 `snapshot()`，OO 初始化打印配置快照（`QD_OO_CONFIG_DIAG`），并补充配置校验单测。
- 2026‑03‑20：v1.6 推进 Phase 2 第二批：`ParamsRegistry.from_env()` 与参数校验/快照落地，`QingdaiWorld.run()` 输出 schema 化 `run_metadata` 并支持写入 `QD_OO_METADATA_JSON`。
- 2026‑03‑20：v1.7 推进 Phase 2 第三批：OO 主循环将 MLD/热容/湿度初值/边界层与生态耦合开关等配置改由 `ParamsRegistry` 驱动，新增参数与 metadata 回归测试。
- 2026‑03‑20：v1.8 推进 Phase 2 第四批：OO 路径将湿度/水文参数改为构造后注入（`HumidityParams/HydrologyParams`），并移除 `snowpack_step` 对 env 的直接读取，补充注入回归测试。
- 2026‑03‑20：v1.9 推进 Phase 2 第五批：`coupler.py` 去除 env getter 默认路径，改为 world 显式注入 `energy/humidity` 参数；新增注入回归测试并完成分批回归。
- 2026‑03‑20：v1.10 推进 Phase 2 第六批：`SimConfig` 扩展运行控制字段并接管 `world.run()` 剩余运行期 env 读取（总时长与 metadata 输出），补充配置映射与路径稳定性回归测试。
- 2026‑03‑20：v1.11 推进 Phase 2 第七批：`SimConfig` 新增 metadata enable 与默认轨道分数策略，`world.run()` 运行控制逻辑全部由配置对象闭环驱动，新增默认时长与开关回归测试。
- 2026‑03‑20：v1.12 推进 Phase 2 第八批：明确 `SimConfig/ParamsRegistry` schema 分组边界并新增 `from_snapshot`，补齐配置与参数的 roundtrip 序列化一致性测试。
- 2026‑03‑21：v1.13 完成 Phase 2 收口：新增 `world/world_step_ops.py` 纯函数流水线并重构 `world.run()` 为薄编排（读-调-写-swap）；补充 `tests/test_world_step_ops.py` 并完成分批回归。
- 2026‑03‑21：v1.14 启动 Phase 3 首批：新增 `OceanOrchestrator` 并接入 `world.run()`（支持 DI 注入），补充注入编排回归测试。
- 2026‑03‑21：v1.15 推进 Phase 3 第二批：`world.run()` 海洋路径统一为 orchestrator 接口并移除 fallback；新增 legacy ocean backend 自动包装回归测试。
- 2026‑03‑21：v1.16 推进 Phase 3 第三批：新增 `AtmosphereOrchestrator`，`world.run()` 大气路径统一为 orchestrator 接口并移除 fallback；新增 injected/legacy atmosphere 回归测试。
- 2026‑03‑21：v1.17 推进 Phase 3 第四批：新增 `orchestrator_spec.py` 并以 `WorldOrchestratorConfigureSpec` 统一 Atmos/Ocean configure 输入，减少 `world.run()` 参数拼装噪音并保持回归稳定。
- 2026‑03‑21：v1.18 完成 Phase 3 收口：新增 `build_world_orchestrator_spec` 下沉 spec 组装逻辑并补充 `tests/test_orchestrator_spec.py`，进一步瘦身 `world.run()`。
- 2026‑03‑21：v1.19 启动 Phase 4 首批：新增 `RoutingOrchestrator`，将 routing 纳入 world 编排并统一 injected/legacy 兼容；`SimConfig` 增加 routing 运行配置字段并补齐回归测试。
- 2026‑03‑21：v1.20 推进 Phase 4 第二批：新增 `HydrologyOrchestrator` 并将 hydrology 从 `world_step_ops` 内收；`world.run()` 固定 `hydrology → routing → ecology` 顺序，新增 injected/legacy hydrology 回归测试。
- 2026‑03‑21：v1.21 推进 Phase 4 第三批：新增 `EcologyOrchestrator` 并内收生态子步/日界调度；`world.run()` 形成 hydrology/routing/ecology 统一 orchestrator 链路并补齐 injected/legacy ecology 回归测试。
- 2026‑03‑21：v1.22 推进 Phase 4 第四批：新增统一 `world_diagnostics` 结构并汇总 energy/water + hydrology/routing/ecology 诊断，补齐统一诊断输出回归测试。
- 2026‑03‑21：v1.23 推进 Phase 4 第五批：为 `world_diagnostics` 增加 schema 版本化校验与可选 JSON 落盘（`QD_OO_WORLD_DIAG_*`），补齐诊断 schema/落盘回归测试。
- 2026‑03‑21：v1.24 推进 Phase 4 第六批：新增 `WorldDiagnosticsDocument` 等 dataclass contract 与 `world_diagnostics_from_dict/make_world_diagnostics_document`，引入严格字段校验与可选 backward-compat 解析策略。
- 2026‑03‑21：v1.25 完成 Phase 4 收口并切换 Phase 5：补齐 Phase 4 验收清单并将项目状态更新为“Phase 5 启动中”。
- 2026‑03‑21：v1.26 完成 Phase 5 收口：完善 JAX 运行策略（`QD_JAX_ALLOW_CPU`）、增强 `benchmark_jax.py` 的内存峰值与 JSON 输出，并补齐 Phase5 回归测试与文档同步。

---

## 附录 A（草案）：World API 编排规范与可插拔数值核

动机  
- 在 P020 的 OO 重构（Façade/DI/JAX‑First/DBA）之上，进一步明确“层引擎（大气/海洋/陆地/生态）—耦合器（辐射/湿度/云/风应力/海表/路由）—端口（Ports）—薄编排（Orchestrator）”的统一契约。  
- 目标：DRY、小文件（每文件 ≲ 300 行）、规范命名与人类知识体系对齐、不同格点/差分/谱方案的无缝接入、DBA 读写语义保持不变。

A. 统一接口（world.api）  
- 层引擎（数组入/出；DBA 由编排器所有）：  
  - AtmosEngine: step(u,v,h,dt,**kw)→(u,v,h)  
  - OceanEngine: step(uo,vo,eta,sst,dt,**kw)→(uo,vo,eta,sst)  
  - LandEngine: step(inputs: dict, dt, **kw)→outputs: dict  
  - EcologyEngine: step(state: dict, dt, **kw)→state_next: dict  
- 耦合器（Ports→fluxes/updates；与学科命名一致）：  
  - RadiationCoupler / HumidityCoupler / CloudCoupler / WindStressCoupler / OceanSurfaceCoupler / RoutingCoupler  
- 工厂（可替换为注册表）：make_atmos_engine / make_coupler  
- 见：pygcm/world/api/__init__.py

B. 端口（Typed Ports）  
- SurfaceToAtmosphere、AtmosphereToSurfaceFluxes、ColumnProcessIn/Out（pygcm/world/ports.py）  
- 显式承载界面（海/陆—气）与体内（q/云/降水）数据流；有利于守恒与类型校验。

C. 薄编排 Orchestrator（以大气为例）  
- 文件：pygcm/world/atmosphere.py（仅 DBA 读写 + 协议调用）  
- 读取 .read → 调用 engine.step(数组) → 写入 .write；不在此处 swap（世界所有）。  
- 调用 coupler.compute(ports) → 得到 (AtmosphereToSurfaceFluxes, ColumnProcessOut)。  
- 同样风格可推广至 Ocean/Land/Ecology 编排器。

示例（简化）  
```python
engine = make_atmos_engine("legacy_spectral", grid=..., friction_map=..., land_mask=..., C_s_map=...)
coupler = make_coupler("default")
atm = Atmosphere(engine=engine, coupler=coupler)

fluxes, col_out = atm.time_step(state, dt,
                                h_eq=None,
                                surface_in=SurfaceToAtmosphere(...),
                                column_in=ColumnProcessIn(...))
state.swap_all()
```

D. 可插拔数值核（Engines）  
- DemoRelaxEngine（pygcm/world/atmos_kernels.py）：线性放松 + 可选弱 ∇⁴，纯数组入/出，便于教学/测试。  
- LegacySpectralBackend（pygcm/world/atmosphere_backend.py）：桥接现有谱/差分混合实现（pygcm.dynamics.SpectralModel）。  
- 两层斜压脚手架（pygcm/world/atmosphere2l.py）：TwoLayerState/Atmosphere2L，示范 2L DBA 与内模耦合路径；可后续定义 TwoLayerEngine 协议使之与 AtmosEngine 对称。

E. 结构化耦合器（Couplers）  
- world/coupler.py：将 Surface/Column 端口组合成界面通量与列内更新，优先使用 energy/humidity 模块；缺席时优雅降级，占位符合守恒方向（例如 LH=Lv·E）。  
- 命名与人类知识体系对齐：Radiation/Humidity/Cloud/WindStress/OceanSurface/Routing。

F. 诊断与守恒  
- world/diagnostics.py：基于 DBA 的 read/write 步级不变量（质量/伪动量/轴向角动量/KE/PE）对比，无需 swap；便于在任何引擎/耦合器组合下做守恒对账。  
- 世界层可统一串联能量/水量闭合（TOA/SFC/ATM、E≈P+R、LH≈LH_release、Qnet 与海表注入一致）。

G. DRY 与小文件  
- 规定：编排器只做“读/调/写”，物理核独立成小文件，引擎/耦合器各司其职；每个文件聚焦单一角色（目标 ≲ 300 行）。  
- 统一接口避免跨文件重复与耦合蔓延，便于后续替换/组合。

H. 与 P020 的关系（升华）  
- 在 P020 的 façade/DI/JAX‑First/DBA 基础上，将“层/因素”抽象沉淀为 world.api 契约，形成面向编排的统一标准；引擎与耦合器在该标准下可自由演进。  
- 任一格点方案（差分/谱/混合）、层数（1L/2L/…）、网格（规则/未来扩展）均可按数组→数组的 step 接口接入，不改编排器与耦合器。

I. 建议的下一步  
- TwoLayerEngine 协议与 2L Orchestrator 对接；Column 端口支持可选 vertical 维度。  
- Ocean/Land/Ecology Engine 的最小协议与示例实现；WindStress/OceanSurface 耦合器示例。  
- 文档：在 docs/12 增补“world.api 契约与编排拓扑”一节（交互图/最小接入指南）。
