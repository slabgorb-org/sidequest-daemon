from unittest.mock import patch, MagicMock

import pytest

from sidequest_daemon.media.r2_writer import upload_pack_asset


def test_upload_pack_asset_writes_to_provided_key():
    fake_client = MagicMock()
    with patch("sidequest_daemon.media.r2_writer._client", return_value=fake_client):
        key = upload_pack_asset(
            r2_key="genre_packs/cav/audio/music/combat.ogg",
            content_bytes=b"fake ogg bytes",
            content_type="audio/ogg",
        )
        assert key == "genre_packs/cav/audio/music/combat.ogg"
        fake_client.put_object.assert_called_once()
        call_kwargs = fake_client.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "genre_packs/cav/audio/music/combat.ogg"
        assert call_kwargs["Body"] == b"fake ogg bytes"
        assert call_kwargs["ContentType"] == "audio/ogg"


def test_upload_pack_asset_rejects_key_outside_genre_packs():
    with pytest.raises(ValueError, match="must start with 'genre_packs/'"):
        upload_pack_asset(
            r2_key="artifacts/foo/bar.ogg",
            content_bytes=b"x",
            content_type="audio/ogg",
        )


def test_download_pack_asset_returns_body_bytes():
    fake_client = MagicMock()
    body = MagicMock()
    body.read.return_value = b"BASE_OGG_BYTES"
    fake_client.get_object.return_value = {"Body": body}
    with patch("sidequest_daemon.media.r2_writer._client", return_value=fake_client):
        from sidequest_daemon.media.r2_writer import download_pack_asset
        data = download_pack_asset("genre_packs/rw/audio/music/convoy.ogg")
        assert data == b"BASE_OGG_BYTES"
        fake_client.get_object.assert_called_once()
        kwargs = fake_client.get_object.call_args.kwargs
        assert kwargs["Key"] == "genre_packs/rw/audio/music/convoy.ogg"


def test_download_pack_asset_rejects_key_outside_genre_packs():
    from sidequest_daemon.media.r2_writer import download_pack_asset
    with pytest.raises(ValueError, match="must start with 'genre_packs/'"):
        download_pack_asset("artifacts/foo/bar.ogg")
