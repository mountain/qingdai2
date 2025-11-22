# AGENTS (Human × AI Engineering Collaboration Handbook)

Purpose and Scope
- This is a reusable handbook for human–AI team collaboration and software engineering, applicable to most Python-centric research/engineering projects.
- It covers roles and responsibilities, task workflow, code and documentation standards, Python best practices, numerics/performance and testability, and collaboration rules under an “agentic” model.
- The examples and patterns are derived from practice (e.g., Config/Params/State separation, Dependency Injection, Façade = API Contract, Pure Kernels + Thin Orchestrators, Double-Buffering state management). They are abstracted for reuse across projects.

1. Roles and Responsibilities (RACI)
1.1 Human Roles
- Project Lead
  - Responsible: Define vision/goals/priorities; P0 decisions; versioning/milestones.
  - Accountable: Approve architectural evolution and major technical direction; enforce risk control and stop-loss mechanisms.
- Domain Experts
  - Consulted/Informed: Provide specialized knowledge (e.g., physics, ecology, finance, healthcare); validate scientific/business correctness; help define parameters and diagnostic acceptance criteria.
- Maintainers/Reviewers
  - Responsible: Code/doc reviews; enforce style, consistency, and quality gates (lint/typing/tests/perf budget/security baseline).
  - Informed: Post-mortems and improvements for CI/CD and release processes.
- Contributors
  - Responsible: Submit incremental changes that meet task standards; comply with templates and checklists.

1.2 AI Roles
- AI Software Engineer
  - Responsible: Design breakdown → implementation → tests → docs; iterate within allowed tools; follow “one-tool-per-step, explicit confirmation, rollbackable” operating contract.
  - Consulted: Offer engineering guidance (boundaries, interface contracts, test matrix, performance and maintainability).
- AI Research Assistant (optional)
  - Consulted: Literature and background research, option comparison, decision memos and risk assessment.
- AI Documentation Engineer (optional)
  - Responsible: Capture decisions and knowledge in docs/ and projects/; keep code–docs–tests aligned and traceable.

2. Agentic Collaboration Methodology
2.1 Task Lifecycle (Plan → Act → Verify → Record)
- Plan
  - Definition of Ready: goals/constraints/acceptance criteria/performance budget/rollback strategy/impact analysis.
  - Outputs: MVP design, interface/signature draft, test points, migration path, risk register.
- Act
  - Principles: Small, reversible steps; favor non-breaking and backward-compatible changes; feature flags + stop-loss.
  - Tool contract (for systems with controlled tools):
    - One tool per step: execute only one tool per message; proceed after success confirmation.
    - Destructive ops (install/remove/network/overwrite) require explicit approval; default deny.
    - Prefer surgical edits (replace_in_file); use full rewrites (write_to_file) for new files or re-layouts only.
    - Read first (list/search/read) → modify → run/verify (execute/devserver/browser) → record.
- Verify
  - Unit/contract/regression/performance/security/doc completeness; meet Definition of Done.
- Record
  - ADRs, change logs, doc chapter updates, project logs (projects/) to ensure knowledge capture and traceability.

2.2 RACI in Tasks
- R: Implementer (human or AI).
- A: Approver accountable for outcomes (lead/maintainer).
- C: Stakeholders to consult (domain experts/upstream-downstream owners).
- I: Stakeholders to inform (ops/data governance/docs).

2.3 Iteration Conventions
- Small, frequent merges; limit PR size (≤ 300–500 changed lines recommended).
- Feature-flag/environment-gated changes; keep legacy path runnable by default (easier rollback).
- Each step leaves a stop-loss point (roll back to last stable state/disable flag).

3. Python Engineering Best Practices (General)
3.1 Packaging and Dependencies
- Use pyproject.toml (PEP 621) for central configuration; lock dependencies (uv/pip-tools/poetry) for reproducibility.
- Define Python version matrix (e.g., 3.11–3.13); CI covers major platforms (Linux/macOS).
- Dependency tiers: runtime, dev (test/lint/typing), optional extras; avoid accidental heavy deps.

3.2 Code Style and Typing
- Unified toolchain: ruff (lint/format/complexity), black (if adopted), isort (imports).
- Typing: mypy/pyright in strict mode; public APIs must be annotated.
- Docstrings: consistent NumPy/Google style; public modules/functions/classes include usage examples and boundaries.

3.3 Configuration Management
- Configuration is determined pre-run and immutable: validate env/.env/.toml via pydantic/pydantic-settings.
- Value validation: ranges/enums/defaults; fail fast on invalid config; freeze config objects to avoid drift.
- Strict separation of “Configuration/Parameters/State” (see §4.1).

Example (Pydantic config model):
```python
from pydantic import BaseModel, field_validator

class AppConfig(BaseModel, frozen=True):
    dt_seconds: float
    filter_type: str
    use_feature_x: bool = True

    @field_validator("dt_seconds")
    @classmethod
    def _dt_pos(cls, v: float):
        assert v > 0
        return v

    @field_validator("filter_type")
    @classmethod
    def _ft_ok(cls, v: str):
        assert v in {"combo", "hyper4", "shapiro", "spectral"}
        return v
```

3.4 Testing Strategy
- Layered tests:
  - Unit: pure functions first; boundary/error paths; fixed random seeds; property tests (hypothesis optional).
  - Contract: interface signature/protocol stability (e.g., façade vs target class reflection checks).
  - Regression: golden outputs (images/arrays with tolerances), invariants (conservation/closure/stats thresholds).
  - Performance: step time/memory/allocations; guardrails in CI smoke benchmarks.
- Coverage targets: core modules 90%+; hot paths must have benchmarks or smoke tests.

3.5 Observability
- Structured logs (JSON/key-value); consistent prefixes and sampling; avoid string concatenation cost on hot paths.
- Diagnostics: key metrics and invariants (energy/mass/closures/stability) at controlled frequency and under toggles.

3.6 Errors and Exceptions
- Clear failure boundaries (input validation, external resources, numerical failures); custom exceptions with context; never swallow stack traces.
- Public API return values and exception semantics are stable; do not break user contracts during compatibility windows.

3.7 I/O and Persistence
- Schema versioning (schema_version); metadata (git_hash, created_at, config snapshot).
- Safe writes: temp file + fsync + atomic replace; rolling backups (N versions).
- Backward compatibility: read old → fill defaults → emit yellow compatibility warnings.

3.8 Performance and Memory
- Vectorize and batch; avoid implicit copies; prefer out= to reuse buffers.
- Minimize lifetime of large objects; prioritize cache hits and incremental updates.
- Optional JAX/NumPy backend: for JAX-first, pure, side-effect-free kernels; handle I/O/object mgmt outside the graph.

4. Architecture and Design (Reusable Patterns)
4.1 Configuration/Parameters/State Separation
- Configuration: set before run, immutable (e.g., dt, switches, backend).
- Parameters: define laws/hyperparameters (constants, coefficients, spectral/physical parameter groups), usually stable within an experiment.
- State: time-evolving fields and caches (arrays/objects), supports snapshots/restarts and schema evolution.

4.2 Dependency Injection (DI) and Factories
- Inject subsystems via constructor for testing and minimal dependencies; offer create_default() factories.
- Allow mocks/stubs for isolated testing.

4.3 Façade = API Contract
- Temporary façade’s public method signatures MUST match the target implementation (names/params/returns).
- Enforce with reflection tests (inspect.signature); enables hot-swapping with zero call-site changes.

4.4 Pure Kernels + Thin Orchestrators
- Implement core computation as pure, stateless, side-effect-free functions with array inputs/outputs; enables @jit/vectorization and unit testing.
- Orchestrators handle only state read/write, kernel calls, and port/flux assembly, not business logic.
- Ports/Adapters define boundaries explicitly; prefer typed ports where feasible.

4.5 Double-Buffering State Management
- Each state field maintains read/write buffers; within a step, read from read and write to write; flip pointers in O(1) at step end.
- Benefits: avoid cross-module dirty reads; reduce allocations/copies; synergizes with JAX (DBA stays outside the graph).
- Implementation: provide a DoubleBufferingArray (DBA) and State.swap_all() atomic flip; forbid swap within subsystems.

Pseudo-code:
```python
class DoubleBufferingArray:
    def __init__(self, shape, dtype=float):
        self._a = np.zeros(shape, dtype=dtype)
        self._b = np.zeros(shape, dtype=dtype)
        self._read_idx = 0
    @property
    def read(self):
        return self._a if self._read_idx == 0 else self._b
    @property
    def write(self):
        return self._b if self._read_idx == 0 else self._a
    def swap(self):
        self._read_idx ^= 1

class WorldState:
    def swap_all(self):
        for field in self._iter_dba_fields():
            field.swap()
```

4.6 Backend Interop (NumPy/JAX)
- Jitted kernels accept only native arrays; DBA never enters the jit graph; outer orchestration writes back via `.write[:] = jit_kernel(dba.read, ...)`.
- Explicitly avoid `__array__` implicit copies on hot paths.

4.7 Compatibility and Migration
- Feature flags/environment gates the new path; keep old path runnable by default.
- Phase migration with stop-loss and perf/conservation regression; update Changelog and docs.

5. Numerics/Data and Invariants (for scientific/engineering projects)
- Determinism and reproducibility: fixed seeds; record dependency versions; floating-point policy (fp64 default; exceptions documented).
- Invariants and conservation: define project “hard constraints” (e.g., energy/mass/closure, stability thresholds); enforce in CI.
- Tolerances and comparisons: arrays via rtol/atol; images via SSIM/PSNR; consistent diagnostic windows.
- Boundaries and stability: required filters/diffusion/regularization; centralized parameter control and logging.

6. Documentation and Knowledge Management
- docs/: developer/user manuals (architecture/module APIs, run configs, numerical stability, FAQ, best practices).
- projects/: project/milestones/design/proposals (timestamps, status, next steps, risks/mitigations).
- ADRs: record architectural decisions and trade-offs across alternatives.
- Changelog: versions, API changes, compatibility, migration guides.
- In-code docs: public API docstrings parseable by doc generators; runnable minimal examples.

7. Security, Release, and Operations
- Security/compliance: no secrets in repo; least privilege; data anonymization; dependency security scans.
- Release: semantic versioning (SemVer); trunk-based or simplified GitFlow; main branch remains releasable.
- CI/CD: lint/type/test/benchmark/artifacts; fail closed; artifacts are traceable (include metadata).
- Monitoring and rollback: release checklists; staged rollout/feature flags; rollback on anomalous signals.

8. Operational Contract for AI Collaboration
- Task inputs must include:
  - Goal; acceptance criteria (functionality/perf/docs/no regression); constraints (interfaces/compatibility/deadlines); risks and rollback strategy.
  - Code path/doc context; run method and data scale; whether network/installs are allowed.
- AI execution rules:
  - Read before write: build context with list/search/read; then modify; then run and verify.
  - One tool per step + wait for confirmation after each tool.
  - Destructive ops require explicit approval; no global installs/deletes/overwrites by default.
  - Prefer surgical edits; provide full contents when creating files; match project style.
  - Prioritize runnability: every change ships with minimal runnable example or tests.
  - Knowledge capture: update docs/ and projects/ upon completion; align with PR checklist.
- Definition of Done requires:
  - All tests passing; new tests cover critical paths; perf/memory/allocation budgets met;
  - Docs/comments/examples complete; backward compatibility or migration guide + flags provided;
  - CI green; security scans pass; changelog updated.

9. Checklists and Templates
9.1 Definition of Ready
- [ ] Goals/scope/non-goals clear; inputs and upstream/downstream identified
- [ ] Acceptance criteria (functionality/perf/docs/regression thresholds) quantifiable
- [ ] Impact analysis (interfaces/data/deployment); risks and rollback
- [ ] Resources and environment (data/permissions/tools) ready
- [ ] Timeline and priority set; RACI clarified

9.2 Definition of Done
- [ ] Unit/contract/regression/performance tests pass (including new cases)
- [ ] Lint/typing pass; public API annotated and documented
- [ ] Docs (docs/) and project logs (projects/) updated
- [ ] Backward-compatible or includes migration with feature flag
- [ ] Production run/release checklist pass; rollback verified

9.3 PR Checklist
- [ ] Small, clear change; commit messages follow convention (Conventional Commits recommended)
- [ ] API signature changes documented in code and Changelog
- [ ] New/changed config has validation and defaults
- [ ] Perf/memory impact explained with benchmarks
- [ ] Security and privacy evaluation (data/deps)

9.4 ADR Template (summary)
- Context → Decision → Alternatives → Trade-offs → Conclusion → Impact/Migration → Rollback

9.5 Task Brief Template (summary)
- Summary; goals/non-goals; inputs/dependencies; acceptance criteria; milestones; risks/rollback; reviewer list

10. Patterns Catalog
- Config/Params/State separation: decouple run-time determinism, scientific parameter sets, and time-evolving state.
- Dependency Injection: inject subsystems; create_default factories; mock-friendly.
- Façade = API Contract: adapt legacy first with matching signatures; zero-change hot swap.
- Pure Kernels + Thin Orchestrators: stateless kernels; orchestration limits to read/write/scheduling; JAX-first friendly.
- Double Buffering (DBA): read/write separation + atomic swap; balances memory/perf and debuggability.
- Feature Flags and Stop-Loss: gate new paths; ensure phased migration safety.
- Schema Versioning: persisted/exported data with schema_version and metadata; migration tooling included.
- Invariant Guards: encode conservation/closures/thresholds in tests and runtime diagnostics.
- Small files and single responsibility: ≤ ~300 lines per file; names align with domain/business semantics.
- Dual-track perf and correctness: guard both; avoid focusing on only one.

11. Anti-patterns (avoid)
- Big-bang rewrites; global replacements without stop-loss.
- Implicit global state and magical side effects; uncontrolled in-place mutation.
- Passing state objects into JIT kernels causing host↔device copies or trace failures.
- Docs drifting from code; changes without tests/benchmarks/migration guidance.
- Ungated breaking changes; no rollback strategy or compatibility promise.
- Giant PRs; API drift merged without contract tests.

12. Quick Reference and Minimal Examples
- Read–write separation and write-back (pseudo-code)
```python
def step(orchestrator, state, params, cfg, dt):
    # READ
    a = state.field_a.read
    b = state.field_b.read
    # PURE KERNELS
    out = compute_tendency(a, b, params)  # side-effect-free array function
    # WRITE
    state.field_a.write[:] = a + out * dt
    # Do not swap here; higher-level layer calls swap_all()
```
- JAX-first interop
```python
@jax.jit
def kernel(arr, p): ...
state.x.write[:] = kernel(state.x.read, params)
```

Appendix: Suggested Tooling (optional)
- Packaging/deps: uv or pip-tools or poetry; pyproject.toml
- Quality: ruff + mypy/pyright + pytest + coverage + hypothesis (optional)
- Docs: mkdocs/sphinx; docstring conventions; runnable examples
- Benchmarks: pytest-benchmark or custom scripts/benchmark_*.py
- Security: pip-audit/safety; pre-commit hooks

Closing
- The goal of this handbook is to turn human intent into high-quality, maintainable, and verifiable engineering assets with minimal friction. Agentic collaboration is not “automate everything,” but a set of standards, contracts, and checklists that combine AI speed with human judgment to produce stable delivery.
- Projects should tailor this handbook to local constraints (team size/complexity/compliance), and feed successful practices back into docs and templates to create a sustainable, evolving collaboration system.
