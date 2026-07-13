import json
import os
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


def _classifier_install_bash_block() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    macos = readme.split("### macOS / Linux", maxsplit=1)[1].split(
        "### Windows", maxsplit=1
    )[0]
    code = macos.split("```bash", maxsplit=1)[1].split("```", maxsplit=1)[0]
    return code.split("cd smart-album-cleaner", maxsplit=1)[1].split(
        "./run.sh", maxsplit=1
    )[0]


def _run_classifier_install_bash_block(
    tmp_path: Path, shasum_exit: int, block: str | None = None
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    event_log = tmp_path / "events.log"
    curl = fake_bin / "curl"
    curl.write_text(
        "#!/bin/sh\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = \"-o\" ]; then\n"
        "    shift\n"
        "    printf 'fake classifier' > \"$1\"\n"
        "    exit 0\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "exit 2\n",
        encoding="utf-8",
    )
    shasum = fake_bin / "shasum"
    shasum.write_text(
        "#!/bin/sh\n"
        "printf 'verify\\n' >> \"$EVENT_LOG\"\n"
        "exit \"$SHASUM_EXIT\"\n",
        encoding="utf-8",
    )
    mv = fake_bin / "mv"
    mv.write_text(
        "#!/bin/sh\n"
        "printf 'move\\n' >> \"$EVENT_LOG\"\n"
        "/bin/mv \"$@\"\n",
        encoding="utf-8",
    )
    mktemp = fake_bin / "mktemp"
    mktemp.write_text(
        "#!/bin/sh\n"
        'temporary="$TMPDIR/classifier.download"\n'
        ': > "$temporary"\n'
        'printf \'%s\\n\' "$temporary"\n',
        encoding="utf-8",
    )
    for command in (curl, shasum, mv, mktemp):
        command.chmod(0o755)
    env = os.environ | {
        "EVENT_LOG": str(event_log),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SHASUM_EXIT": str(shasum_exit),
        "TMPDIR": str(tmp_path),
    }
    completed = subprocess.run(
        ["bash", "-c", block or _classifier_install_bash_block()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    events = event_log.read_text(encoding="utf-8").splitlines()
    return completed, events


def _assert_classifier_install_failure(
    tmp_path: Path, completed: subprocess.CompletedProcess[str], events: list[str]
) -> None:
    assert completed.returncode != 0
    assert events == ["verify"]
    assert not (tmp_path / "models/mobilenet_face.pth").exists()
    assert not (tmp_path / "classifier.download").exists()


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


def test_model_card_states_evidence_and_limits():
    card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    for heading in (
        "## Intended Use",
        "## Data",
        "## Evaluation",
        "## Class-level Results",
        "## Limitations and Bias",
        "## Human Review and Recovery",
    ):
        assert heading in card
    assert "75.1%" in card
    assert "subjective" in card.casefold()
    assert "Trash" in card
    assert "not identity recognition" in card.casefold()


def test_readme_does_not_make_unqualified_speed_claims():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "60 秒內" not in readme
    assert "1000 張 < 2 分鐘" not in readme


def test_readme_summarizes_model_evidence_and_limits():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "75.1% accuracy on 193 labelled test examples" in readme
    assert "Performance depends on hardware and image size" in readme
    assert "[Model Card](docs/MODEL_CARD.md)" in readme


def test_model_card_records_verified_release_artifact():
    card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")

    assert "Release tag: `v1.0.0`" in card
    assert (
        "SHA-256: `b7dd3b7d95c13c07167d269d65f49367d0b0007fcf0dc272ab1f94c34f3f4bf0`"
        in card
    )


def test_classifier_release_download_is_pinned_and_verified():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    release_url = (
        "https://github.com/Hunter20041004/smart-album-cleaner/"
        "releases/download/v1.0.0/mobilenet_face.pth"
    )
    digest = "b7dd3b7d95c13c07167d269d65f49367d0b0007fcf0dc272ab1f94c34f3f4bf0"
    macos = readme.split("### macOS / Linux", maxsplit=1)[1].split(
        "### Windows", maxsplit=1
    )[0]
    windows = readme.split("### Windows", maxsplit=1)[1].split(
        "## 開發模式", maxsplit=1
    )[0]

    assert "releases/latest/download/mobilenet_face.pth" not in readme
    assert readme.count(release_url) == 2
    assert readme.count(digest) == 2
    assert 'MODEL_TMP="$(mktemp)"' in macos
    assert "shasum -a 256 -c -" in macos
    assert macos.index("curl --fail") < macos.index("shasum -a 256 -c -")
    assert macos.index("shasum -a 256 -c -") < macos.index(
        'mv "$MODEL_TMP" models/mobilenet_face.pth'
    )
    assert "%TEMP%" in windows
    assert "Get-FileHash" in windows
    assert windows.index("curl -fL") < windows.index("Get-FileHash")
    assert windows.index("Get-FileHash") < windows.index("Move-Item")
    assert "Size: `12,098,423 bytes`" in card


def test_classifier_install_bash_block_fails_closed_before_move(tmp_path):
    failed, failed_events = _run_classifier_install_bash_block(
        tmp_path / "failed", shasum_exit=1
    )
    succeeded, succeeded_events = _run_classifier_install_bash_block(
        tmp_path / "succeeded", shasum_exit=0
    )

    _assert_classifier_install_failure(tmp_path / "failed", failed, failed_events)
    assert succeeded.returncode == 0
    assert succeeded_events == ["verify", "move"]
    assert (tmp_path / "succeeded/models/mobilenet_face.pth").is_file()


def test_classifier_install_validator_rejects_missing_temp_cleanup(tmp_path):
    cleanup_trap = 'trap \'rm -f "$MODEL_TMP"\' EXIT\n'
    block = _classifier_install_bash_block()
    assert cleanup_trap in block
    hostile_block = block.replace(cleanup_trap, "", 1)
    failed, failed_events = _run_classifier_install_bash_block(
        tmp_path, shasum_exit=1, block=hostile_block
    )
    assert (tmp_path / "classifier.download").is_file()

    with pytest.raises(AssertionError):
        _assert_classifier_install_failure(tmp_path, failed, failed_events)


def test_windows_classifier_verification_has_fail_closed_control_flow():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    windows = readme.split("### Windows", maxsplit=1)[1].split(
        "## 開發模式", maxsplit=1
    )[0]
    after_verification = windows.split("powershell -NoProfile", maxsplit=1)[1]
    guard = "if errorlevel 1 exit /b 1"

    assert guard in after_verification
    assert after_verification.index(guard) < after_verification.index(
        "python -m venv"
    )


def _validate_model_card_against_report(card: str, report: str) -> None:
    report_test_count = re.search(r"^Test n:\s*(\d+)$", report, re.MULTILINE)
    report_threshold = re.search(
        r"^Threshold \(P\(Bad\)\):\s*(\d+\.\d+)$", report, re.MULTILINE
    )
    card_test_count = re.search(r"test set of (\d+) examples", card)
    card_threshold = re.search(r"decision threshold of (\d+\.\d+)", card)
    report_metrics = {
        match.group(1): match.groups()[1:]
        for match in re.finditer(
            r"^\s*(Bad|Good)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+"
            r"(\d+\.\d+)\s+(\d+)$",
            report,
            re.MULTILINE,
        )
    }
    card_metrics = {
        match.group(1): match.groups()[1:]
        for match in re.finditer(
            r"^\| `(Bad|Good)` \| (\d+\.\d+) \| (\d+\.\d+) \| "
            r"(\d+\.\d+) \| (\d+) \|$",
            card,
            re.MULTILINE,
        )
    }
    data_limits = " ".join(
        card.split("## Data", maxsplit=1)[1]
        .split("## Evaluation", maxsplit=1)[0]
        .casefold()
        .split()
    )
    recovery = " ".join(
        card.split("## Human Review and Recovery", maxsplit=1)[1]
        .split("## Model Artifact Integrity", maxsplit=1)[0]
        .casefold()
        .split()
    )

    assert report_test_count is not None
    assert report_threshold is not None
    assert card_test_count is not None
    assert card_threshold is not None
    assert card_test_count.group(1) == report_test_count.group(1) == "193"
    assert card_threshold.group(1) == report_threshold.group(1)
    assert set(report_metrics) == {"Bad", "Good"}
    assert card_metrics == report_metrics
    provenance_limits = re.search(
        r"does not contain a tracked measured report that establishes (.+?)\. "
        r"those facts are unknown\.",
        data_limits,
    )
    assert provenance_limits is not None
    not_documented_facts = provenance_limits.group(1)
    for fact in ("dataset's origin", "contributor consent", "licence"):
        assert fact in not_documented_facts
    assert "split independence therefore cannot be confirmed" in data_limits
    assert "irreversible deletion without an explicit human decision" in recovery


def test_model_card_matches_tracked_report_and_states_safety_limits():
    report = (ROOT / "reports/classification_report.txt").read_text(
        encoding="utf-8"
    )
    card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")

    _validate_model_card_against_report(card, report)


def test_model_card_validator_rejects_known_claims_masked_by_split_unknown():
    report = (ROOT / "reports/classification_report.txt").read_text(
        encoding="utf-8"
    )
    card = (ROOT / "docs/MODEL_CARD.md").read_text(encoding="utf-8")
    before_data, after_data = card.split("## Data", maxsplit=1)
    _, after_evaluation = after_data.split("## Evaluation", maxsplit=1)
    hostile_card = (
        f"{before_data}## Data\n\n"
        "The tracked report identifies a test set of 193 examples. "
        "The dataset origin is known. Contributor consent is confirmed. "
        "The dataset licence is known. Split independence therefore cannot "
        "be confirmed and remains unknown.\n\n"
        f"## Evaluation{after_evaluation}"
    )

    with pytest.raises(AssertionError):
        _validate_model_card_against_report(hostile_card, report)
