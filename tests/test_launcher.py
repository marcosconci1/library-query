import os
import stat
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parent.parent / "library-query.sh"


def test_launcher_does_not_inherit_foreign_python_environment(tmp_path):
    poisoned = tmp_path / "poison"
    poisoned.mkdir()
    (poisoned / "json.py").write_text("raise SystemExit('poisoned import')\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "env.txt"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'PYTHONPATH=%s\\nPYTHONHOME=%s\\n' \"${{PYTHONPATH-unset}}\" \"${{PYTHONHOME-unset}}\" > '{record}'\n"
    )
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "PYTHONPATH": str(poisoned), "PYTHONHOME": str(poisoned)}
    result = subprocess.run(["bash", str(LAUNCHER), "--help"], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert record.read_text().splitlines() == ["PYTHONPATH=unset", "PYTHONHOME=unset"]
