from types import SimpleNamespace

from src.face_detector import FaceBox, boxes_from_result


def test_tasks_result_is_converted_to_absolute_face_boxes():
    result = SimpleNamespace(
        detections=[
            SimpleNamespace(
                bounding_box=SimpleNamespace(origin_x=12, origin_y=34, width=56, height=78),
            )
        ]
    )

    assert boxes_from_result(result) == [FaceBox(x=12, y=34, width=56, height=78)]


def test_tasks_result_without_detections_returns_empty_list():
    assert boxes_from_result(SimpleNamespace(detections=[])) == []
