import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from services.demo_data import demo_manifest, seed_demo_data, verify_video


class DemoSeedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / 'backend'
        self.data = Path(self.temp.name) / 'data'
        self.folder = self.backend / 'snaps' / 'demo-job'
        (self.folder / 'faces').mkdir(parents=True)
        (self.folder / 'faces' / 'person_0.png').write_bytes(b'thumbnail')
        (self.folder / 'detection_metadata.json').write_text(json.dumps({'unique_faces': [{'person_id': 'person_0', 'snap_filename': 'person_0.png'}]}))
        (self.folder / 'job_manifest.json').write_text(json.dumps({'job_id': 'demo-job', 'twelvelabs_video_id': 'video-a', 'video_path': '/old/computer/clip.mp4', 'video_filename': 'clip.mp4', 'status': 'ready'}))
        (self.backend / 'output').mkdir()
        self.source = self.backend / 'output' / 'clip.mp4'
        self.source.write_bytes(b'demo-source')
        self.entry = {'video_id': 'video-a', 'job_id': 'demo-job', 'source_file': 'clip.mp4', 'source_size': 11, 'source_sha256': hashlib.sha256(b'demo-source').hexdigest()}
        (self.backend / 'demo').mkdir()
        (self.backend / 'demo' / 'catalog.json').write_text(json.dumps({'videos': [self.entry]}))

    def test_seed_installs_thumbnails_and_rebases_exact_source(self):
        self.assertEqual(seed_demo_data(self.backend, self.data), 1)
        saved = self.data / 'snaps' / 'demo-job'
        manifest = json.loads((saved / 'job_manifest.json').read_text())
        self.assertEqual(manifest['twelvelabs_video_id'], 'video-a')
        self.assertEqual(manifest['video_path'], str(self.source.resolve()))
        self.assertEqual((saved / 'faces' / 'person_0.png').read_bytes(), b'thumbnail')

    def test_redeployment_preserves_existing_job_and_metadata(self):
        seed_demo_data(self.backend, self.data)
        saved = self.data / 'snaps' / 'demo-job' / 'detection_metadata.json'
        saved.write_text('{"user_updated": true}')
        self.assertEqual(seed_demo_data(self.backend, self.data), 0)
        self.assertEqual(saved.read_text(), '{"user_updated": true}')

    def test_new_detection_run_takes_precedence(self):
        newer = self.data / 'snaps' / 'new-detection'
        newer.mkdir(parents=True)
        (newer / 'job_manifest.json').write_text(json.dumps({'twelvelabs_video_id': 'video-a', 'status': 'processing'}))
        self.assertEqual(seed_demo_data(self.backend, self.data), 0)
        self.assertFalse((self.data / 'snaps' / 'demo-job').exists())

    def test_wrong_video_binding_fails_validation(self):
        wrong = {**self.entry, 'video_id': 'another-video'}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            demo_manifest(self.backend, wrong)

    def test_wrong_source_content_is_rejected(self):
        self.assertTrue(verify_video(self.source, self.entry))
        self.source.write_bytes(b'wrong-video')
        self.assertFalse(verify_video(self.source, self.entry))

    def test_missing_face_thumbnail_is_rejected(self):
        (self.folder / 'faces' / 'person_0.png').unlink()
        with self.assertRaisesRegex(ValueError, 'snapshot missing'):
            demo_manifest(self.backend, self.entry)


if __name__ == '__main__':
    unittest.main()
