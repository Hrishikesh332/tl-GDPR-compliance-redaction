from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from services import pipeline


class VideoSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.output_patch = patch.object(pipeline, 'OUTPUT_DIR', str(self.output))
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)

    def test_exact_demo_job_source_is_reused_for_detection(self):
        source = self.output / 'demo-source.mp4'
        source.write_bytes(b'video')
        with patch.object(pipeline, 'get_exact_job_id_by_video_id', return_value='demo-job'), patch.object(pipeline, 'load_job_manifest', return_value={'video_path': str(source)}):
            self.assertEqual(pipeline.infer_video_path_for_video('demo-video', info={}), str(source))

    def test_other_video_is_never_used_as_fallback(self):
        (self.output / 'unrelated.mp4').write_bytes(b'video')
        with patch.object(pipeline, 'get_exact_job_id_by_video_id', return_value=None):
            self.assertIsNone(pipeline.infer_video_path_for_video('new-video', info={'system_metadata': {'filename': 'missing.mp4'}}))

    def test_exact_filename_with_wrong_duration_is_not_reused(self):
        (self.output / 'clip.mp4').write_bytes(b'video')
        with patch.object(pipeline, 'get_exact_job_id_by_video_id', return_value=None), patch.object(pipeline, 'get_video_metadata', return_value={'duration_sec': 30}):
            self.assertIsNone(pipeline.infer_video_path_for_video('new-video', info={'system_metadata': {'filename': 'clip.mp4', 'duration': 90}}))

    def test_hls_download_does_not_reuse_another_videos_filename(self):
        (self.output / 'clip.mp4').write_bytes(b'unrelated')
        def download(command, **kwargs):
            Path(command[-1]).write_bytes(b'downloaded')
            from types import SimpleNamespace
            return SimpleNamespace(returncode=0)
        with patch.object(pipeline.subprocess, 'run', side_effect=download) as run:
            result = pipeline.download_video_from_hls('https://example.test/stream.m3u8', 'new-video', filename='clip.mp4')
            self.assertEqual(result, str(self.output / 'input_new-video.mp4'))
            run.assert_called_once()
            self.assertEqual((self.output / 'clip.mp4').read_bytes(), b'unrelated')

    def test_new_video_still_downloads_and_starts_detection(self):
        source = self.output / 'downloaded.mp4'
        source.write_bytes(b'video')
        with patch.object(pipeline, 'get_exact_job_id_by_video_id', return_value=None), patch.object(pipeline, 'infer_video_path_for_video', return_value=None), patch.object(pipeline.twelvelabs_service, 'get_video_info', return_value={'hls': {'video_url': 'https://example.test/stream.m3u8'}, 'system_metadata': {'filename': 'new.mp4'}}), patch.object(pipeline, 'download_video_from_hls', return_value=str(source)) as download, patch.object(pipeline, 'start_ingestion', return_value='new-job') as start:
            self.assertEqual(pipeline.ensure_job_for_video('new-video'), 'new-job')
            download.assert_called_once_with('https://example.test/stream.m3u8', 'new-video', filename='new.mp4')
            self.assertEqual(start.call_args.kwargs['existing_video_id'], 'new-video')
            self.assertTrue(start.call_args.kwargs['skip_indexing'])


if __name__ == '__main__':
    unittest.main()
