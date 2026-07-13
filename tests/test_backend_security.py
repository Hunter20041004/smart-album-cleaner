from pathlib import Path

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
