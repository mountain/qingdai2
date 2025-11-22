# DBA 最佳实践（DoubleBufferingArray Best Practices）

状态（2025‑09‑28）
- [x] 初版（结合 world/state 与 world/atmosphere 实作经验）
- [ ] 扩充：数值核与网格度量（∇²/∇⁴ 的网格一致化、极区处理）
- [ ] 扩充：端到端集成案例（World → Atmosphere/Ocean/Hydro → swap_all → 守恒回归）

本文档总结在 Qingdai GCM 中应用 DoubleBufferingArray（DBA）的约定与最佳实践，目的是在保持高可读性与 JAX-friendly 的同时，最大化性能收益与工程稳健性。所有示例均可直接对照以下实现：
- world/state：DBA 包装的 WorldState 与 swap_all 原子翻转
- world/atmosphere：纯核（pure kernels）+ 薄编排（thin orchestrator）的 DBA 写法
- tests/test_world_state.py：DBA 读/写与 swap 行为测试

---

## 1. 设计目标

- 可读性：模块内业务逻辑清晰，读/写边界一目了然，不混淆「当前态」与「下一态」。
- JAX 兼容：纯核函数用于 @jit；DBA 不进入 jitted 函数图，避免 host↔device 复制与 trace 失败。
- 性能友好：消除热点路径的重复分配与复制，步末统一 O(1) 指针翻转（swap_all）。
- 可测试：DBA 行为（读/写隔离、swap 生效）与物理核（数值正确性）各自独立测试。

---

## 2. 基本约定

1) 纯函数内核（pure kernels）
- 输入/输出均为原生数组（numpy/jax），不接触 DBA。
- 禁止全局状态与隐式副作用；允许通过 out= 参数写出。
- 内核命名建议：xxx_tendency/xxx_update/xxx_kernel 等。

2) 薄编排层（orchestrator）
- 仅在此层使用 DBA。
- 从 `.read` 读取当前态；向 `.write[:]` 写入下一态。
- 严禁在编排层修改 `.read` 或执行 swap；swap_all 归属于「世界/驱动」层。

3) JAX 互操作
- jitted 函数仅接收/返回原生数组；DBA 永不进入 @jit。
- 将 jitted 输出（或 out= 写入结果）在编排层通过 `.write[:] = ...` 回写。

4) 一致的时序
- 每步内：各子系统仅写入 `.write` 缓冲；步末世界对象统一 `swap_all()`。
- 禁止子系统内部提前 swap 或跨模块混合读写。

---

## 3. 代码骨架（参考）

纯核（示例）：
```python
def momentum_relaxation_tendency(u: Array, v: Array, tau_s: float) -> tuple[Array, Array]:
    inv_tau = 0.0 if tau_s <= 0 else (1.0 / tau_s)
    return -u * inv_tau, -v * inv_tau

def height_relaxation_tendency(h: Array, h_eq: Array | float | None, tau_s: float) -> Array:
    inv_tau = 0.0 if tau_s <= 0 else (1.0 / tau_s)
    target = 0.0 if h_eq is None else h_eq
    return -(h - target) * inv_tau
```

编排（示例）：
```python
def time_step(self, state, dt: float, *, h_eq: Array | None = None) -> None:
    # READ
    u = state.atmos.u.read
    v = state.atmos.v.read
    h = state.atmos.h.read

    du_dt, dv_dt = momentum_relaxation_tendency(u, v, self.params.tau_relax_u_s)
    dh_dt = height_relaxation_tendency(h, h_eq, self.params.tau_relax_h_s)

    # WRITE
    state.atmos.u.write[:] = u + du_dt * dt
    state.atmos.v.write[:] = v + dv_dt * dt
    state.atmos.h.write[:] = h + dh_dt * dt
    # 注：不在此处 swap；世界步末统一 swap_all()
```

世界层（步末统一）：
```python
world.atmos.time_step(world.state, dt)
# world.ocean.time_step(...)
# world.hydro.time_step(...)
world.state.swap_all()
```

---

## 4. 单元测试建议

- 读/写隔离 + 生效语义（tests/test_world_state.py 已示范）
  - 写 `.write[:]` 前后，`.read` 不变；`swap_all()` 后 `.read` 反映写入内容。
- 纯核正确性（无需 DBA）
  - 用小网格、确定性输入测试数值输出；可并行覆盖 numpy/jax（若开启 JAX）。
- 编排层契约
  - 断言函数不触碰 `.read`；仅对 `.write` 切片赋值；
  - 不调用 swap；不在函数中保留对 `.write`/`.read` 的可变闭包引用。

---

## 5. 命名与结构

- State 命名：`AtmosState/SurfaceState/OceanState/HydroState` 等，字段全为 `DBA`。
- 构造器：`zeros_world_state(shape) / zeros_world_state_from_grid(grid)`。
- 纯核模块建议集中在 `pygcm/<domain>_core.py` 或当前模块顶部，便于 @jit。
- 编排类：`Atmosphere/Surface/Hydrology/...`，接受 `Params` 数据类作为配置。

---

## 6. 性能要点

- 禁止隐式 `__array__` 触发拷贝（例如 `np.sin(dba)`）：
  - 热点路径显式 `arr = dba.read` 再传入内核。
- 大型表达式优先使用 `out=` 避免中间数组：
  - `np.multiply(a, b, out=out); np.add(out, c, out=out)`。
- JAX 路径下：
  - jitted 内核只处理原生数组；外层回写 `.write[:]`。
  - 避免在步内多次 trace；必要时加 warmup。

---

## 7. 诊断与日志

- 诊断矩阵与时间序列应单独存储（或使用 side-channel），避免污染当前/下一态缓冲。
- 频率控制：每 N 步打印/汇总，减少 I/O 与格式化开销。

---

## 8. 常见陷阱与防护

- 陷阱：在编排层中对 `.read` 原地写（无切片）。
  - 防护：代码审查 + 小测试；静态检查可加黑名单 API 搜索。
- 陷阱：子系统内部执行 swap，导致跨模块读写乱序。
  - 防护：规定 swap 仅由世界层调用；在子系统层添加注释与断言。
- 陷阱：将 DBA 传入 @jit 内核，导致 host ↔ device 复制或 trace 失败。
  - 防护：内核签名只接收原生数组；将 DBA 访问限制在编排层。

---

## 9. 与 JAX 的互操作（简表）

| 场景 | 建议写法 |
| --- | --- |
| 纯核 | `@jit` 可包裹；入出均为原生数组 |
| 编排 | 不使用 `@jit`；读 `.read`，写 `.write[:]` |
| 混合 | 外层编排先 `arr = dba.read`；`out = jit_kernel(arr, ...)`；回写 `dba.write[:] = out` |

---

## 10. 扩展示例：超扩散倾向（演示版）

此处提供一个简化 ∇⁴ 演示核（见 `world/atmosphere.py`），用于占位与结构说明。实际生产实现需：
- 使用球面/网格一致化的拉普拉斯算子（含 cosφ 与极区处理）；
- 支持周期经度与极点环平均/一致化；
- 以 `out=` 复用缓冲减少中间数组。

---

## 11. 迁移策略

- 自内向外：先在模块内部切换到「纯核 + 薄编排」，再用 WorldState DBA 接线，最后由世界步末统一 swap。
- 分阶段替换子系统：Atmosphere → Ocean → Hydrology → Routing → Ecology。
- 每阶段加入回归（能量/水量闭合、步时基准），失败即回退。

---

## 12. 快速检查清单（开发者）

- [ ] 纯核与编排职责清晰，DBA 仅出现在编排层。
- [ ] 所有写入均为 `.write[:] = ...` 切片赋值。
- [ ] 任何函数中未调用 swap，swap_all 仅在世界层。
- [ ] 纯核可直接用于 numpy/jax（必要时加 @jit）。
- [ ] 单元测试覆盖「读/写隔离 + swap 生效」与内核正确性。
- [ ] 热点路径避免隐式 `__array__` 触发拷贝。

---

## 参考实现

- world/state.py：DBA 包装的 WorldState 与构造器、swap_all
- world/atmosphere.py：DBA 风格 Atmosphere（纯核 + 薄编排）
- tests/test_world_state.py：DBA 行为测试
- scripts/benchmark_double_buffering.py：DB vs Naive 基准脚本

如需贡献更多模块的 DBA 化改造，请按本文约定提交 PR，并附带最小单测与（可选）基准对照。
