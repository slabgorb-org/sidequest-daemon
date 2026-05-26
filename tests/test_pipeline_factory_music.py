import asyncio
from sidequest_daemon.media.pipeline_factory import MediaPipelineFactory
from sidequest_daemon.media.music_pipeline import MusicPipeline


def test_factory_constructs_music_pipeline():
    factory = MediaPipelineFactory()
    factory.init_music(render_lock=asyncio.Lock())
    assert isinstance(factory.music_pipeline, MusicPipeline)


def test_factory_wires_r2_downloader():
    """The constructed pipeline must carry a callable downloader so audio2audio
    refs can be fetched (wiring test — proves the factory connects it)."""
    factory = MediaPipelineFactory()
    factory.init_music(render_lock=asyncio.Lock())
    assert callable(factory.music_pipeline._r2_downloader)
