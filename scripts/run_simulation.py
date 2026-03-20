"""
Main simulation entrypoint (OO-only runtime).
"""

from __future__ import annotations

import os
from contextlib import suppress

from pygcm.jax_compat import is_enabled as JAX_IS_ENABLED
from pygcm.world import QingdaiWorld


def main() -> None:
    print("--- Initializing Qingdai GCM ---")
    with suppress(Exception):
        print(
            f"[JAX] Acceleration enabled: {JAX_IS_ENABLED()} "
            "(toggle via QD_USE_JAX=1; platform via QD_JAX_PLATFORM=cpu|gpu|tpu)"
        )
    try:
        oo_strict = int(os.getenv("QD_USE_OO_STRICT", "0")) == 1
    except Exception:
        oo_strict = False
    world = QingdaiWorld.create_default()
    print("[P020] QingdaiWorld OO orchestrator active (Phase 5).")
    if oo_strict:
        world.run(n_steps=1)
        print("[P020] QD_USE_OO_STRICT=1 → exiting after OO smoke run.")
        return
    world.run()


if __name__ == "__main__":
    main()
