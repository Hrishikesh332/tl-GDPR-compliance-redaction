import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services import twelvelabs_service


class IndexInfoTests(unittest.TestCase):
    def retrieve(self, index):
        client = Mock()
        client.indexes.retrieve.return_value = index
        with patch.object(twelvelabs_service, "get_client", return_value=client):
            result = twelvelabs_service.get_index_info(index_id="test-index")
        client.indexes.retrieve.assert_called_once_with(index_id="test-index")
        return result

    def test_current_sdk_fields_keep_existing_api_contract(self):
        index = SimpleNamespace(
            id="test-index", index_name="Redaction", video_count=2,
            total_duration=45.0, created_at="2026-01-01T00:00:00Z", updated_at=None,
            models=[SimpleNamespace(model_name="marengo3.0", model_options=["visual", "audio"])],
        )
        result = self.retrieve(index)
        self.assertEqual(result["name"], "Redaction")
        self.assertEqual(result["models"], [{"name": "marengo3.0", "options": ["visual", "audio"]}])
        self.assertEqual(result["video_count"], 2)
        self.assertEqual(result["total_duration"], 45.0)
        self.assertEqual(result["created_at"], "2026-01-01T00:00:00Z")
        self.assertIsNone(result["updated_at"])

    def test_legacy_sdk_fields_still_work(self):
        result = self.retrieve(SimpleNamespace(id="test-index", name="Legacy", models=[SimpleNamespace(name="marengo", options=["visual"])], video_count=0))
        self.assertEqual(result["name"], "Legacy")
        self.assertEqual(result["models"], [{"name": "marengo", "options": ["visual"]}])
        self.assertEqual(result["video_count"], 0)

    def test_missing_optional_fields(self):
        result = self.retrieve(SimpleNamespace(index_name="Empty", models=None))
        self.assertEqual(result["index_id"], "test-index")
        self.assertEqual(result["models"], [])
        self.assertIsNone(result["created_at"])


if __name__ == "__main__":
    unittest.main()
