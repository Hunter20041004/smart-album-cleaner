from pathlib import Path
from unittest.mock import MagicMock, patch

from src import predict_face


def test_prediction_loader_requests_weights_only(tmp_path):
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"checkpoint")
    checkpoint = {
        "state_dict": {},
        "classes": ["Good", "Bad"],
        "decision_threshold": 0.6,
        "arch": "mobilenet_v2",
    }
    model = MagicMock()
    model.to.return_value = model

    predict_face._model_cache.clear()
    with patch.object(predict_face, "_pick_device", return_value="cpu"), patch.object(
        predict_face.torch, "load", return_value=checkpoint,
    ) as load, patch.object(predict_face, "build_model", return_value=model):
        predict_face._load_model(model_path)

    load.assert_called_once_with(model_path, map_location="cpu", weights_only=True)


def test_all_project_checkpoint_loads_use_weights_only():
    root = Path(__file__).resolve().parents[1]
    for relative in ("src/predict_face.py", "src/evaluate.py", "src/train_mobilenet.py"):
        source = (root / relative).read_text(encoding="utf-8")
        assert "weights_only=False" not in source
