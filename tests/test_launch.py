import os
from pathlib import Path

from evalbench import launch


def test_launch_command_watches_src_and_app():
    cmd = launch.build_command()
    joined = " ".join(cmd)
    assert "streamlit" in joined and cmd[cmd.index("run") + 1].endswith("streamlit_app.py")
    watch = cmd[cmd.index("--server.folderWatchList") + 1]
    assert Path(watch).is_absolute() and Path(watch).name == "src"
    assert "--browser.gatherUsageStats" in cmd


def test_launch_env_puts_src_on_pythonpath():
    # Streamlit only hot-reloads modules under the app folder or on PYTHONPATH
    env = launch.build_env({"PYTHONPATH": "C:/other"})
    parts = env["PYTHONPATH"].split(os.pathsep)
    assert Path(parts[0]).name == "src" and "C:/other" in parts
