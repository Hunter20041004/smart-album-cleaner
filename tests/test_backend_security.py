from pathlib import Path
import json
import subprocess

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import main

client = TestClient(main.app, base_url="http://localhost")


@pytest.fixture(autouse=True)
def clear_authorized_roots():
    main.AUTHORIZED_ROOTS.clear()
    yield
    main.AUTHORIZED_ROOTS.clear()


def test_authorized_path_allows_file_inside_selected_root(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "photo.jpg"
    photo.write_bytes(b"photo")

    main.authorize_root(album)

    assert main.require_authorized_path(photo) == photo.resolve()


def test_authorized_path_rejects_sibling_file_without_leaking_path(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private")
    main.authorize_root(album)

    with pytest.raises(HTTPException) as exc:
        main.require_authorized_path(outside)

    assert exc.value.status_code == 403
    assert str(outside) not in str(exc.value.detail)


def test_authorized_path_rejects_symlink_escape(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    outside = tmp_path / "private.jpg"
    outside.write_bytes(b"private")
    link = album / "link.jpg"
    link.symlink_to(outside)
    main.authorize_root(album)

    with pytest.raises(HTTPException) as exc:
        main.require_authorized_path(link)

    assert exc.value.status_code == 403


def test_scan_rejects_paths_that_were_not_selected_and_authorized(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "private.jpg"
    photo.write_bytes(b"private")

    with pytest.raises(HTTPException) as exc:
        main.start_scan(main.ScanRequest(paths=[str(photo)]))

    assert exc.value.status_code == 403
    assert main.AUTHORIZED_ROOTS == set()


def test_native_folder_picker_authorizes_only_the_selected_root(tmp_path, monkeypatch):
    album = tmp_path / "album"
    album.mkdir()
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout=f"{album}/\n", stderr=""
        ),
    )

    result = main.select_folder()

    assert result == {"cancelled": False, "folder": str(album.resolve())}
    assert main.AUTHORIZED_ROOTS == {album.resolve()}


def test_native_file_picker_authorizes_the_selected_files_common_root(
    tmp_path, monkeypatch
):
    album = tmp_path / "album"
    album.mkdir()
    first = album / "first.jpg"
    second = album / "second.png"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    monkeypatch.setattr(main.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout=f"{first}\n{second}\n",
            stderr="",
        ),
    )

    result = main.select_files()

    assert result == {
        "cancelled": False,
        "paths": [str(first.resolve()), str(second.resolve())],
    }
    assert main.AUTHORIZED_ROOTS == {album.resolve()}


def test_launch_files_never_bind_to_all_network_interfaces():
    root = Path(__file__).resolve().parents[1]
    launch_text = (root / "run.sh").read_text() + (root / "README.md").read_text()
    assert "--host 0.0.0.0" not in launch_text


def test_cross_site_browser_origin_is_rejected():
    response = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 403


def test_documented_loopback_origin_is_allowed():
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_non_loopback_host_is_rejected():
    response = client.get("/api/health", headers={"Host": "attacker.example"})
    assert response.status_code == 400


def test_temp_album_preview_trash_and_restore_flow(tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "photo.jpg"
    photo.write_bytes(b"synthetic fixture")
    main.authorize_root(album)

    preview = main.get_image(str(photo), w=0)
    moved = main.move_to_trash(main.TrashRequest(folder=str(album), paths=[str(photo)]))
    trash_path = Path(main._read_manifest(album)[0]["trash_path"])
    restored = main.restore_from_trash(
        main.RestoreRequest(folder=str(album), trash_paths=[str(trash_path)])
    )

    assert Path(preview.path).name == "photo.jpg"
    assert moved["moved"] == [str(photo)]
    assert restored["restored"] == 1
    assert photo.read_bytes() == b"synthetic fixture"


def test_restore_empty_selection_is_noop(tmp_path, monkeypatch):
    album = tmp_path / "album"
    trash = album / "Trash"
    trash.mkdir(parents=True)
    trashed_photo = trash / "photo.jpg"
    trashed_photo.write_bytes(b"synthetic fixture")
    original_photo = album / "photo.jpg"
    main.authorize_root(album)
    main._write_manifest(
        album,
        [{
            "original_path": str(original_photo),
            "trash_path": str(trashed_photo),
            "deleted_at": "2026-07-13T00:00:00",
        }],
    )
    manifest = main._manifest_path(album)
    manifest_before = manifest.read_bytes()
    move_calls = []
    monkeypatch.setattr(
        main.shutil,
        "move",
        lambda *args: move_calls.append(args),
    )

    response = client.post(
        "/api/trash/restore",
        json={"folder": str(album), "trash_paths": []},
    )

    assert response.status_code == 200
    assert response.json() == {"restored": 0, "failed": [], "remaining": 1}
    assert manifest.read_bytes() == manifest_before
    assert move_calls == []


def test_system_trash_empty_selection_is_noop(tmp_path, monkeypatch):
    album = tmp_path / "album"
    trash = album / "Trash"
    trash.mkdir(parents=True)
    trashed_photo = trash / "photo.jpg"
    trashed_photo.write_bytes(b"synthetic fixture")
    original_photo = album / "photo.jpg"
    main.authorize_root(album)
    main._write_manifest(
        album,
        [{
            "original_path": str(original_photo),
            "trash_path": str(trashed_photo),
            "deleted_at": "2026-07-13T00:00:00",
        }],
    )
    manifest = main._manifest_path(album)
    manifest_before = manifest.read_bytes()
    system_trash_calls = []
    monkeypatch.setattr(
        main,
        "_move_file_to_system_trash",
        lambda path: system_trash_calls.append(path),
    )

    response = client.post(
        "/api/trash/system-delete",
        json={"folder": str(album), "trash_paths": []},
    )

    assert response.status_code == 200
    assert response.json() == {"deleted": 0, "failed": [], "remaining": 1}
    assert manifest.read_bytes() == manifest_before
    assert system_trash_calls == []


def test_restore_failure_does_not_expose_exception_details(tmp_path, monkeypatch):
    album = tmp_path / "album"
    trash = album / "Trash"
    trash.mkdir(parents=True)
    trashed_photo = trash / "photo.jpg"
    trashed_photo.write_bytes(b"synthetic fixture")
    original_photo = album / "photo.jpg"
    main.authorize_root(album)
    main._write_manifest(
        album,
        [{
            "original_path": str(original_photo),
            "trash_path": str(trashed_photo),
            "deleted_at": "2026-07-13T00:00:00",
        }],
    )
    monkeypatch.setattr(
        main.shutil,
        "move",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("filesystem credential secret-value")
        ),
    )

    result = main.restore_from_trash(
        main.RestoreRequest(
            folder=str(album), trash_paths=[str(trashed_photo)]
        )
    )

    assert result["failed"] == ["photo.jpg: 還原失敗"]
    assert "secret-value" not in json.dumps(result)


def test_move_to_trash_failure_does_not_expose_exception_details(
    tmp_path, monkeypatch
):
    album = tmp_path / "album"
    album.mkdir()
    photo = album / "photo.jpg"
    photo.write_bytes(b"synthetic fixture")
    main.authorize_root(album)
    monkeypatch.setattr(
        main.shutil,
        "move",
        lambda *args: (_ for _ in ()).throw(
            RuntimeError("filesystem credential secret-value")
        ),
    )

    result = main.move_to_trash(
        main.TrashRequest(folder=str(album), paths=[str(photo)])
    )

    assert result["failed"] == ["photo.jpg: 移至 Trash 失敗"]
    assert "secret-value" not in json.dumps(result)


def test_system_trash_failure_does_not_expose_exception_details(
    tmp_path, monkeypatch
):
    album = tmp_path / "album"
    trash = album / "Trash"
    trash.mkdir(parents=True)
    trashed_photo = trash / "photo.jpg"
    trashed_photo.write_bytes(b"synthetic fixture")
    original_photo = album / "photo.jpg"
    main.authorize_root(album)
    main._write_manifest(
        album,
        [{
            "original_path": str(original_photo),
            "trash_path": str(trashed_photo),
            "deleted_at": "2026-07-13T00:00:00",
        }],
    )
    monkeypatch.setattr(
        main,
        "_move_file_to_system_trash",
        lambda path: (_ for _ in ()).throw(
            RuntimeError("Finder credential secret-value")
        ),
    )

    result = main.move_trash_items_to_system_trash(
        main.SystemTrashRequest(
            folder=str(album), trash_paths=[str(trashed_photo)]
        )
    )

    assert result["failed"] == ["photo.jpg: 移至系統垃圾桶失敗"]
    assert "secret-value" not in json.dumps(result)


def test_trash_directory_symlink_cannot_escape_authorized_root(tmp_path):
    album = tmp_path / "album"
    outside = tmp_path / "outside"
    album.mkdir()
    outside.mkdir()
    photo = album / "photo.jpg"
    photo.write_bytes(b"private")
    (album / "Trash").symlink_to(outside, target_is_directory=True)
    main.authorize_root(album)

    with pytest.raises(HTTPException) as exc:
        main.move_to_trash(main.TrashRequest(folder=str(album), paths=[str(photo)]))

    assert exc.value.status_code == 403
    assert photo.is_file()
    assert not (outside / "photo.jpg").exists()
