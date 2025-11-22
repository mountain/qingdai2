# AGENTS（人类 × AI 工程协作手册）

目的与范围
- 本文是可复用的“人类—AI 团队协作与工程实践”手册，适用于多数以 Python 为主的科研/工程项目。
- 内容覆盖角色分工、任务流转、代码与文档标准、Python 最佳实践、数值/性能与可测试性、以及“Agentic”模式下的协作准则。
- 文中示例与模式来自实践沉淀（如：配置/参数/状态三分法、依赖注入、Façade＝API 合约、纯核＋薄编排、双缓冲状态管理等），已抽象为通用做法，可移植到其他项目。


1. 角色与职责（RACI）
1.1 人类角色
- 项目负责人（Project Lead）
  - R（Responsible）制定愿景/目标/优先级；P0 决策；版本/里程碑把控。
  - A（Accountable）批准架构演进与重大技术路线；风险受控与止损机制。
- 领域专家（Domain Experts）
  - C/I：提供专业知识（物理/生态/金融/医学等域）；校验科学与业务正确性；参与参数与诊断验收标准制定。
- 维护者/评审者（Maintainers/Reviewers）
  - R：代码/文档评审；守护风格、一致性与质量闸门（lint/typing/tests/性能预算/安全基线）。
  - I：CI/CD 管线与发布流程事后回顾与改进。
- 贡献者（Contributors）
  - R：按任务标准提交增量变更；遵守约定的模板与清单。

1.2 AI 角色
- AI 软件工程师（AI Software Engineer）
  - R：方案拆解—实现—测试—文档；在工具许可范围内进行迭代；遵守“单步单工具—显式确认—可回滚”的操作契约。
  - C：提出工程化建议（边界约束、接口合约、测试矩阵、性能与可维护性）。
- AI 研究助手（可选）
  - C：文献/资料梳理、备选方案调研对比、给出决策备忘与风险评估。
- AI 文档工程师（可选）
  - R：将决策与知识固化到 docs/ 与 projects/；保持“代码—文档—测试”三位一体的可追溯性。


2. Agentic 协作方法论
2.1 任务生命周期（Plan → Act → Verify → Record）
- Plan（规划）
  - 输入充足性（Definition of Ready）：目标/约束/验收标准/性能预算/回退策略/影响范围。
  - 设计输出：最小可行方案（MVP）、接口/签名草案、测试点、迁移路径与风险清单。
- Act（实施）
  - 原则：小步快跑、可回滚；优先非破坏性与向后兼容；“门控开关 + 止损点”。
  - 工具契约（适用于多数具备“受控工具”的 AI 系统）：
    - 单步单工具：每条消息只执行一次工具；下一步在确认成功后再继续。
    - 破坏性操作（安装/删除/网络/覆盖）需显式批准；安全默认拒绝。
    - 文件修改优先精准编辑（replace_in_file），全量重写（write_to_file）仅在创建新文件或重构重排时使用。
    - 读取优先（read/search/list）→ 修改 → 运行/验证（execute/devserver/browser）→ 记录。
- Verify（验证）
  - 单元/契约/回归/性能/安全/文档完整性检查；满足 Definition of Done。
- Record（记录）
  - 设计与决策（ADR）、变更说明（Changelog）、文档章节更新、项目记录（projects/），保证知识沉淀与可追溯。

2.2 RACI 在任务中的配置
- R：负责实现的直接执行者（可为 AI/人）。
- A：对结果负责的审批者（通常为项目负责人/维护者）。
- C：应被征询的相关方（领域专家/上下游模块维护者）。
- I：需被告知的干系人（运维/数据治理/文档）。

2.3 迭代惯例
- 小而频繁的合并；限制 PR 体积（≤ 300–500 行变更为宜）。
- 变更以 Feature Flag/环境变量门控；默认保持旧路径可运行（回退更容易）。
- 每一步都留下“止损点”（可回滚至上一个稳定状态/关闭开关）。


3. Python 工程最佳实践（通用）
3.1 包与依赖
- 使用 pyproject.toml（PEP 621）集中配置；锁定依赖（例如 uv/pip-tools/poetry 任一）以实现可复现构建。
- Python 版本矩阵明确（例如 3.11–3.13）；CI 覆盖主流平台（Linux/macOS）。
- 依赖分层：runtime、dev（test/lint/typing）、optional extras；避免意外的重量依赖。

3.2 代码风格与类型
- 统一工具链：ruff（lint/格式化/复杂度）、black（如项目采用）、isort（导入排序）。
- 类型检查：mypy/pyright 打开严格模式；公共 API 必须注解。
- 文档字符串：NumPy/Google 风格一致化；公共模块/函数/类提供用法示例与边界说明。

3.3 配置管理
- 配置（Configuration）在运行前确定且不可变：使用 pydantic/pydantic-settings 校验 env/.env/.toml。
- 值验证：范围/枚举/默认；失败早抛（fail fast）；配置对象冻结（frozen）避免运行时漂移。
- 将“配置/参数/状态”严格区分（见 §4.1）。

示例（Pydantic 配置模型）：
```python
from pydantic import BaseModel, field_validator

class AppConfig(BaseModel, frozen=True):
    dt_seconds: float
    filter_type: str
    use_feature_x: bool = True

    @field_validator("dt_seconds")
    @classmethod
    def _dt_pos(cls, v: float): assert v > 0; return v

    @field_validator("filter_type")
    @classmethod
    def _ft_ok(cls, v: str):
        assert v in {"combo", "hyper4", "shapiro", "spectral"}
        return v
```

3.4 测试策略
- 层次化测试：
  - 单元测试：纯函数优先、边界与异常路径、随机种子固定、属性测试（hypothesis 可选）。
  - 契约测试：接口签名/协议一致性（例如 façade 与目标类反射比对）。
  - 回归测试：黄金结果（图像/数组容差）、不变量（守恒/闭合/统计阈值）。
  - 性能基准：步时/内存/分配次数；防回退（CI 中烟测基准）。
- 覆盖率目标：核心模块 90%+；热路径必须有基准或烟测。

3.5 可观测性
- 结构化日志（JSON/键值对）；统一前缀与采样频率；热路径避免字符串拼接开销。
- 诊断面板：关键指标与不变量（能量/质量/闭合/稳定性）定期输出；开关可控。

3.6 错误与异常
- 明确错误边界（输入校验、外部资源、数值失败）；自定义异常类型，补充上下文；避免吞掉栈信息。
- 对外 API 返回值与异常语义稳定；兼容期内不可破坏用户契约。

3.7 I/O 与持久化
- Schema 版本化（schema_version）；元数据记录（git_hash、创建时间、配置快照）。
- 安全写：临时文件 + fsync + 原子替换；滚动备份 N 份。
- 兼容旧格式：读旧→填默认→打印兼容警告（黄色）。

3.8 性能与内存
- 向量化与批处理；避免隐式复制；优先 out= 写法重用缓冲。
- 大对象生命周期最小化；缓存命中与增量计算优先。
- JAX/NumPy 后端可选：JAX-first 时核心核函数纯函数化、无副作用；图外进行 I/O 与对象管理。


4. 体系结构与代码设计（可复用模式）
4.1 配置/参数/状态（Configuration/Parameters/State）三分法
- Configuration：运行前确定，不随时间变化（如步长 dt、开关、后端）。
- Parameters：定义规律或实验超参数（如常数、系数、光谱/物理参数组），通常一试验内稳定。
- State：随时间演化的场与缓存（数组/对象），可支持快照/重启与 schema 演进。

4.2 依赖注入（DI）与工厂
- 通过构造函数注入子系统，便于测试替换与最小依赖；提供 create_default() 工厂组装默认实现。
- Mock/Stub 可注入，构建隔离测试。

4.3 Façade ＝ API 合约
- 临时 façade 的公开方法签名必须与目标实现一致（方法名/参数/返回类型）。
- 通过反射测试（inspect.signature）守护接口稳定；实现“热切换零改动”。

4.4 纯核 + 薄编排（Pure Kernels & Thin Orchestrators）
- 核心计算尽量做成纯函数（stateless，无副作用），入/出均为数组；方便 @jit/向量化与单元测试。
- 编排器仅负责读/写状态、调用核函数与组装端口/通量，不做复杂业务逻辑。
- 端口（Ports）/适配器（Adapters）明确边界：输入输出结构化、可类型化（Typed Ports）。

4.5 双缓冲状态管理（Double Buffering）
- 每个状态场维护 read/write 双缓冲；单步只读 read、只写 write；步末 O(1) 交换指针（swap）。
- 优点：避免跨模块“脏读”；显著降低分配与复制；与 JAX 互补（DBA 留在图外）。
- 实现建议：提供 DoubleBufferingArray（DBA）与 State.swap_all() 原子翻转；禁止在子系统内 swap。

伪代码示例：
```python
class DoubleBufferingArray:
    def __init__(self, shape, dtype=float):
        self._a = np.zeros(shape, dtype=dtype)
        self._b = np.zeros(shape, dtype=dtype)
        self._read_idx = 0
    @property
    def read(self):  return self._a if self._read_idx == 0 else self._b
    @property
    def write(self): return self._b if self._read_idx == 0 else self._a
    def swap(self): self._read_idx ^= 1

class WorldState:
    def swap_all(self):
        for field in self._iter_dba_fields():
            field.swap()
```

4.6 后端互操作（NumPy/JAX）
- jitted 核心仅接受原生数组；DBA 不进入图；外层编排 `.write[:] = jit_kernel(dba.read, ...)`。
- 热路径显式避免 `__array__` 隐式拷贝。

4.7 版本兼容与迁移
- Feature Flag/Env 门控新路径；默认保持旧路径可用。
- 分阶段迁移，有“止损点”与基准/守恒回归；变更附带 Changelog 与文档更新。


5. 数值/数据与不变量（适配科学/工程类项目）
- 确定性与可复现：固定随机种子；记录外部依赖版本；浮点精度策略（fp64 默认，特殊场景例外）。
- 不变量与守恒：定义项目内“硬约束”（如能量/水量/质量闭合、数值稳定阈值）；CI 中加入阈值验证。
- 容差与比较：数组比较使用相对/绝对容差；图像使用 SSIM/PSNR；诊断时间窗一致。
- 边界与稳定：必要的滤波/扩散/正则化；参数集中配置并记录。


6. 文档与知识管理
- docs/：面向开发者与用户的手册（架构/模块 API、运行配置、数值稳定、FAQ、最佳实践等）。
- projects/：项目/里程碑/设计/提案（含时间戳、状态、下一步、风险与对策）。
- ADR（Architecture Decision Record）：记录架构决策与备选方案权衡。
- 变更记录（Changelog）：版本、接口变更、兼容性、迁移指南。
- 代码内文档：公共 API 的 docstring 必须可被文档生成工具解析；示例最小可运行。


7. 安全、发布与运维
- 安全与合规：密钥不入库；最小权限访问；数据脱敏；依赖安全扫描。
- 发布：语义化版本（SemVer）；trunk-based 开发或简化 GitFlow；常青分支保持可发布。
- CI/CD：lint/type/test/基准/发布工件；失败即阻断；产物可溯源（包含元数据）。
- 监控与回滚：发布检查表，灰度/开关控制；异常指标触发回退。


8. 与 AI 协作的操作契约（Operational Contract）
- 任务输入需要包含：
  - 目标、验收标准（功能/性能/文档/回归不回退）、约束（接口/兼容/截止时间）、风险与回退策略。
  - 代码路径/文档上下文、运行方式与数据规模、是否允许网络或安装依赖。
- AI 执行准则：
  - 先读后写：优先使用读取类工具（list/search/read）建立上下文；再修改；再运行验证。
  - 单步单工具 + 等待确认：每次工具执行后，等待结果与确认再进入下一步。
  - 破坏性操作需显式批准；默认不进行全局安装/删除/覆盖。
  - 修改文件优先精准替换；创建新文件需提供完整内容；兼顾项目风格。
  - 可运行性优先：任何改动都附带最小可运行示例或测试。
  - 知识沉淀：完成后更新 docs/ 与 projects/，同步 PR 模板中的核对项。
- 交付物（Definition of Done）必须满足：
  - 所有测试通过；新增测试覆盖关键路径；性能/内存/分配预算满足；
  - 文档/注释/示例齐备；向后兼容（或提供迁移指南与开关）；
  - CI 绿色；安全扫描通过；变更记录更新。


9. 清单与模板
9.1 Definition of Ready（任务就绪）
- [ ] 目标/范围/非目标明确；输入与上下游识别清楚
- [ ] 验收标准（功能/性能/文档/回归阈值）可量化
- [ ] 影响面评估（接口/数据/部署）；风险与回退策略
- [ ] 资源与环境（数据/权限/工具）就绪
- [ ] 工期与优先级确定；RACI 明确

9.2 Definition of Done（完成）
- [ ] 单元/契约/回归/性能测试通过（含新加用例）
- [ ] Lint/Typing 通过；公共 API 注释补齐
- [ ] 文档（docs/）与项目记录（projects/）更新
- [ ] 向后兼容或附迁移方案与 Feature Flag
- [ ] 产线运行/发布检查清单通过；可回滚方案验证

9.3 PR 检查表
- [ ] 变更小而清晰；提交信息遵循约定（Conventional Commits 推荐）
- [ ] 接口签名变化已在文档与 Changelog 记录
- [ ] 新/变更配置带有校验与默认值
- [ ] 性能/内存影响有说明与基准数据
- [ ] 安全与隐私评估（数据/依赖）

9.4 ADR 模板（摘要）
- 背景 → 决策 → 备选方案 → 权衡 → 结论 → 影响面与迁移 → 回退方案

9.5 任务简档模板（摘要）
- 摘要、目标/非目标、输入/依赖、验收标准、里程碑、风险与回退、评审人列表


10. 可复用模式目录（Patterns Catalog）
- Config/Params/State 三分法：将运行时确定性、科学参数集与时变状态分离。
- 依赖注入（DI）：构造时注入子系统；默认工厂 create_default；Mock 友好。
- Façade = API 合约：先以 façade 适配旧实现，签名与目标实现一致，热切换零改动。
- 纯核 + 薄编排：stateless 内核；编排只读写与调度；可 JAX-first。
- 双缓冲（DBA）：读写分离 + 步末原子 swap；内存/性能与可调试性兼顾。
- 门控与止损：Feature Flag/环境变量门控；阶段化迁移的安全阀。
- Schema 版本化：持久化/导出含 schema_version 与元数据；带迁移工具。
- 不变量守护：将守恒/闭合/阈值纳入测试与运行诊断。
- 小文件/单一职责：每文件 ≲ 300 行为宜；命名对齐学科/业务语义。
- 基准与回归双轨：性能与 correctness 同步守护，避免只顾一端。


11. 反模式（应避免）
- 大爆炸式重写；无止损点的全局替换。
- 隐式全局状态与魔法副作用；不可控的 in-place 修改。
- 将状态对象直接传入 JIT 内核导致 host↔device 复制与 trace 失败。
- 文档与代码脱节；变更缺少测试/基准/迁移指南。
- 未经门控的破坏性变更；缺失回退策略与兼容承诺。
- 巨型 PR；接口漂移不经合约测试即合入。


12. 速查与最小示例
- 读写分离与回写（伪代码）
```python
def step(orchestrator, state, params, cfg, dt):
    # READ
    a = state.field_a.read
    b = state.field_b.read
    # PURE KERNELS
    out = compute_tendency(a, b, params)  # 无副作用数组函数
    # WRITE
    state.field_a.write[:] = a + out * dt
    # 不在此处 swap；由更高层统一 swap_all()
```
- JAX-first 互操作
```python
@jax.jit
def kernel(arr, p): ...
state.x.write[:] = kernel(state.x.read, params)
```

附：建议工具链（可选）
- 包/依赖：uv 或 pip-tools 或 poetry；pyproject.toml
- 质量：ruff + mypy/pyright + pytest + coverage + hypothesis（可选）
- 文档：mkdocs/sphinx；docstring 规范；示例可运行
- 基准：pytest-benchmark 或自定义 scripts/benchmark_*.py
- 安全：pip-audit/safety；pre-commit 钩子


结语
- 本手册的价值在于“让人类意图以最小摩擦转化为高质量、可维护、可验证的工程资产”。Agentic 协作并非“自动化一切”，而是通过标准、合同与清单，将 AI 的速度与人类的判断力合成稳定的交付能力。
- 任何项目都应根据实际边界条件（团队规模/复杂度/合规要求）裁剪本手册，并在实践中将成功经验反馈到文档与模板中，形成可持续演进的协作体系。
