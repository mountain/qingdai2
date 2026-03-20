import numpy as np
from pygcm.numerics.double_buffer import DoubleBufferingArray as DBA


def test_read_write_swap():
    d = DBA((4, 5), dtype=np.float64, initial_value=1.0)
    r0 = np.array(d.read)
    d.write[:] = d.read + 2.0
    r1 = np.array(d.read)
    assert np.all(r1 == r0)
    d.swap()
    r2 = np.array(d.read)
    assert np.all(r2 == r0 + 2.0)


def test_array_ufunc_out():
    d = DBA((3, 3), dtype=np.float64, initial_value=0.0)
    np.add(d, 1.0, out=d)
    d.swap()
    assert np.all(np.array(d.read) == 1.0)


def test_self_aliasing_write_raises():
    d = DBA((2, 2), dtype=np.float64, initial_value=0.0)
    raised = False
    try:
        d[:] = d
    except ValueError:
        raised = True
    assert raised


def test_debug_readonly_flag():
    import os

    old = os.getenv("QD_DBA_DEBUG_READONLY")
    os.environ["QD_DBA_DEBUG_READONLY"] = "1"
    try:
        d = DBA((2, 2), dtype=np.float64, initial_value=0.0)
        arr = d.read
        wrote = False
        try:
            arr[...] = 1.0
            wrote = True
        except Exception:
            wrote = False
        assert wrote is False
    finally:
        if old is None:
            os.environ.pop("QD_DBA_DEBUG_READONLY", None)
        else:
            os.environ["QD_DBA_DEBUG_READONLY"] = old
