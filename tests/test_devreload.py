import sys
import types

from evalbench import devreload


def test_refresh_purges_package_modules_when_sources_change(tmp_path):
    src = tmp_path / "pkg"
    src.mkdir()
    f = src / "a.py"
    f.write_text("x = 1\n")
    state = {}
    sys.modules["fakepkg"] = types.ModuleType("fakepkg")
    sys.modules["fakepkg.sub"] = types.ModuleType("fakepkg.sub")
    sys.modules["fakepkgother"] = types.ModuleType("fakepkgother")
    try:
        assert devreload.refresh(state, src, "fakepkg") is False  # first call only records the stamp
        assert "fakepkg.sub" in sys.modules
        f.write_text("x = 22\n")
        assert devreload.refresh(state, src, "fakepkg") is True
        assert "fakepkg" not in sys.modules and "fakepkg.sub" not in sys.modules
        assert "fakepkgother" in sys.modules  # only the package and its submodules
        assert devreload.refresh(state, src, "fakepkg") is False  # unchanged since
    finally:
        for name in ("fakepkg", "fakepkg.sub", "fakepkgother"):
            sys.modules.pop(name, None)
