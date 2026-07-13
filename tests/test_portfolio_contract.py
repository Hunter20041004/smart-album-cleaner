import json
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI

from backend.main import app as backend_app

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ERROR_OUTPUTS = [
    ("sample-001", "Good", "Bad", "89.2", "0.892"),
    ("sample-002", "Bad", "Good", "86.1", "0.139"),
    ("sample-003", "Bad", "Good", "78.3", "0.217"),
    ("sample-004", "Good", "Bad", "74.9", "0.749"),
    ("sample-005", "Good", "Bad", "72.9", "0.729"),
    ("sample-006", "Bad", "Good", "72.8", "0.272"),
    ("sample-007", "Good", "Bad", "70.8", "0.708"),
    ("sample-008", "Good", "Bad", "70.5", "0.705"),
    ("sample-009", "Good", "Bad", "69.5", "0.695"),
    ("sample-010", "Bad", "Good", "68.4", "0.316"),
]
EXPECTED_ERROR_IDS = [output[0] for output in EXPECTED_ERROR_OUTPUTS]
ERROR_REPORT_HEADER = "Top 10 「最自信卻錯了」的測試樣本 — 看這些可以理解模型瓶頸"
ERROR_REPORT_ROW = re.compile(
    r"\s*\d+\.\s+(sample-\d{3})\s+"
    r"\[真實:(Good|Bad)\s*→\s*預測:(Good|Bad)\s*\]"
    r"\s+信心\s+(\d+\.\d)%\s+P\(Bad\)=(\d+\.\d+)"
)


def _validate_error_report(report: str) -> list[tuple[str, ...]]:
    assert "/Users/" not in report
    assert "\\Users\\" not in report
    assert not re.search(r"\bIMG[_-]?\d+", report, re.IGNORECASE)
    assert not re.search(
        r"\.(?:jpe?g|png|heic|webp|bmp|gif|tiff?)\b", report, re.IGNORECASE
    )
    assert "/" not in report and "\\" not in report

    outputs = []
    for line in report.splitlines():
        if not line.strip():
            continue
        if line == ERROR_REPORT_HEADER or re.fullmatch(r"─+", line):
            continue
        match = ERROR_REPORT_ROW.fullmatch(line)
        assert match is not None
        outputs.append(match.groups())
    assert [output[0] for output in outputs] == EXPECTED_ERROR_IDS
    return outputs


def test_canonical_fastapi_vue_entrypoints_are_wired():
    backend_entrypoint = ROOT / "backend" / "main.py"
    frontend_manifest = ROOT / "frontend" / "package.json"
    run_script = ROOT / "run.sh"

    assert backend_entrypoint.is_file()
    assert frontend_manifest.is_file()
    assert run_script.is_file()

    package = json.loads(frontend_manifest.read_text(encoding="utf-8"))
    assert "vue" in package["dependencies"]

    assert isinstance(backend_app, FastAPI)
    assert "/api/health" in {route.path for route in backend_app.routes}

    launch_commands = [
        shlex.split(line)
        for line in run_script.read_text(encoding="utf-8").splitlines()
        if line.lstrip().startswith("exec ")
    ]
    assert any(
        command[:3] == ["exec", ".venv/bin/uvicorn", "backend.main:app"]
        for command in launch_commands
    )


def test_readme_states_current_macos_ui_limit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "原生資料夾／檔案選擇器與完整 UI 掃描流程目前僅支援 macOS" in readme
    assert "Windows 指令僅供前後端開發啟動" in readme


def test_package_metadata_targets_macos():
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]

    assert "Operating System :: OS Independent" not in project["classifiers"]
    assert "Operating System :: MacOS :: MacOS X" in project["classifiers"]


def test_readme_installs_verification_prerequisites_before_checks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    verification = readme.split("## 驗證", maxsplit=1)[1]
    install = ".venv/bin/python -m pip install -r requirements-dev.txt"
    pytest = ".venv/bin/python -m pytest -q"
    ruff = ".venv/bin/python -m ruff check ."

    assert install in verification
    assert verification.index(install) < verification.index(pytest)
    assert verification.index(install) < verification.index(ruff)


def test_no_streamlit_configuration_is_tracked():
    tracked_streamlit_files = subprocess.run(
        ["git", "ls-files", ".streamlit"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert tracked_streamlit_files == []


def test_self_training_commands_use_the_project_virtualenv():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    commands = (
        ".venv/bin/python -m src.prepare_dataset",
        ".venv/bin/python -m src.train_mobilenet --arch mobilenet_v3_large",
        ".venv/bin/python -m src.train_mobilenet --arch mobilenet_v3_large --finetune",
        ".venv/bin/python -m src.evaluate --model models/mobilenet_face.pth",
    )

    assert all(command in readme for command in commands)
    assert "\npython -m src." not in readme


def test_fastapi_vue_is_the_only_current_application():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "streamlit" not in pyproject.casefold()
    assert not (ROOT / "app.py").exists()
    assert "FastAPI + Vue 3" in readme
    assert "MediaPipe 0.10.9" not in readme


def test_install_manifest_has_no_streamlit_runtime():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "streamlit" not in requirements.casefold()


def test_error_report_contains_only_anonymous_sample_ids():
    report = (ROOT / "reports" / "top_errors.txt").read_text(encoding="utf-8")

    assert _validate_error_report(report) == EXPECTED_ERROR_OUTPUTS


def test_error_report_rejects_image_basename_line():
    report = (ROOT / "reports" / "top_errors.txt").read_text(encoding="utf-8")

    with pytest.raises(AssertionError):
        _validate_error_report(f"{report}\nD 壞 (15).jpg")


def test_error_report_rejects_extra_sample_id():
    report = (ROOT / "reports" / "top_errors.txt").read_text(encoding="utf-8")
    extra_row = (
        "11. sample-011 [真實:Good → 預測:Bad ] "
        "信心  50.0%  P(Bad)=0.500"
    )

    with pytest.raises(AssertionError):
        _validate_error_report(f"{report}\n{extra_row}")


def test_error_report_rejects_duplicate_sample_id():
    report = (ROOT / "reports" / "top_errors.txt").read_text(encoding="utf-8")
    duplicate_row = (
        "11. sample-001 [真實:Good → 預測:Bad ] "
        "信心  89.2%  P(Bad)=0.892"
    )

    with pytest.raises(AssertionError):
        _validate_error_report(f"{report}\n{duplicate_row}")
