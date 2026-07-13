import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_fastapi_vue_entrypoints_are_wired():
    backend_entrypoint = ROOT / "backend" / "main.py"
    frontend_manifest = ROOT / "frontend" / "package.json"
    run_script = ROOT / "run.sh"

    assert backend_entrypoint.is_file()
    assert frontend_manifest.is_file()
    assert run_script.is_file()

    package = json.loads(frontend_manifest.read_text(encoding="utf-8"))
    assert "vue" in package["dependencies"]
    assert "backend.main:app" in run_script.read_text(encoding="utf-8")


def test_readme_states_current_macos_ui_limit():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "原生資料夾／檔案選擇器與完整 UI 掃描流程目前僅支援 macOS" in readme
    assert "Windows 指令僅供前後端開發啟動" in readme


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
