import unittest
from unittest.mock import patch

from services import twelvelabs_service as service


class CustomAnalysisTests(unittest.TestCase):
    def test_open_ended_prompt_uses_pegasus_15_and_the_selected_video_source(self):
        prompt = 'Explain how the scene changes and why, with timestamps.'
        source = 'https://video.example.test/selected/playlist.m3u8'
        with patch.object(service, 'get_video_info', return_value={'hls': {'video_url': source}}) as info, patch.object(service, 'twelvelabs_api_request', return_value={'id': 'analysis-123', 'data': '**00:12** The scene changes.'}) as api:
            result = service.analyze_video_custom('indexed-video-123', prompt)
        info.assert_called_once_with('indexed-video-123')
        self.assertEqual(api.call_args.args, ('POST', 'analyze'))
        body = api.call_args.kwargs['json_body']
        self.assertEqual(body['model_name'], 'pegasus1.5')
        self.assertEqual(body['video'], {'type': 'url', 'url': source})
        self.assertIn(prompt, body['prompt'])
        self.assertFalse(body['stream'])
        self.assertEqual(result['data'], '**00:12** The scene changes.')
        self.assertEqual(result['model_name'], 'pegasus1.5')

    def test_source_asset_is_preferred_over_hls_and_indexed_video_id(self):
        info = {'asset_id': 'source-asset-456', 'hls': {'video_url': 'https://example.test/stream.m3u8'}}
        with patch.object(service, 'get_video_info', return_value=info), patch.object(service, 'twelvelabs_api_request', return_value={'data': 'A video answer.'}) as api:
            service.analyze_video_custom('indexed-video-123', 'What happens?')
        self.assertEqual(api.call_args.kwargs['json_body']['video'], {'type': 'asset_id', 'asset_id': 'source-asset-456'})

    def test_video_info_preserves_source_asset_from_sdk(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        client = Mock()
        client.indexes.videos.retrieve.return_value = SimpleNamespace(
            id='indexed-video-123', asset_id='source-asset-456', system_metadata=None,
            user_metadata={}, hls=None, created_at=None, updated_at=None, indexed_at=None,
        )
        with patch.object(service, 'get_client', return_value=client):
            info = service.get_video_info('indexed-video-123', index_id='index-123')
        self.assertEqual(info['asset_id'], 'source-asset-456')
        self.assertEqual(info['video_id'], 'indexed-video-123')

    def test_missing_stream_does_not_submit_an_unrelated_asset(self):
        with patch.object(service, 'get_video_info', return_value={'hls': None}), patch.object(service, 'twelvelabs_api_request') as api:
            with self.assertRaisesRegex(ValueError, 'source is not ready'):
                service.analyze_video_custom('indexed-video-123', 'Describe this video')
        api.assert_not_called()

    def test_empty_generation_surfaces_an_error(self):
        with patch.object(service, 'get_video_info', return_value={'hls': {'video_url': 'https://example.test/clip.m3u8'}}), patch.object(service, 'twelvelabs_api_request', return_value={'data': ''}):
            with self.assertRaisesRegex(RuntimeError, 'returned no analysis'):
                service.analyze_video_custom('video', 'Describe this video')

    def test_upstream_timeout_remains_visible_to_route_error_handling(self):
        with patch.object(service, 'get_video_info', return_value={'hls': {'video_url': 'https://example.test/clip.m3u8'}}), patch.object(service, 'twelvelabs_api_request', side_effect=TimeoutError('Analysis timed out')):
            with self.assertRaises(TimeoutError):
                service.analyze_video_custom('video', 'Describe this video')


if __name__ == '__main__':
    unittest.main()
