import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "check_release.py"


def check(path):
    return subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)


def test_clean_wheel_passes_release_boundary(tmp_path):
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ecam_mv/__init__.py", "__version__ = 'test'\n")
    result = check(wheel)
    assert result.returncode == 0
    assert "release boundary verified" in result.stdout


def test_private_data_cannot_enter_wheel(tmp_path):
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("datasets/school_hourly_weather.csv", "not for the release\n")
    result = check(wheel)
    assert result.returncode == 2
    assert "Repository-only" in result.stderr


def test_archived_prototype_cannot_enter_source_distribution(tmp_path):
    source = tmp_path / "example.tar.gz"
    with tarfile.open(source, "w:gz") as archive:
        data = b"not the modern package"
        member = tarfile.TarInfo("example/legacy_v4/src/ecam_resurrected/__init__.py")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    result = check(source)
    assert result.returncode == 2
    assert "legacy_v4" in result.stderr
