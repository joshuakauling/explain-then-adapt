import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from explain_then_adapt.data_generation.backends.gemini import (
    get_batch,
    parse_gemini_batch_record,
    to_gemini_batch_record,
)
from explain_then_adapt.data_generation.backends.vllm import to_vllm_messages
from explain_then_adapt.data_generation.records import (
    ChatMessage,
    GenerationRequest,
    GenerationStage,
    SamplingParameters,
)


class BackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = GenerationRequest(
            request_id="initial-task-id",
            task_id="task",
            stage=GenerationStage.INITIAL,
            messages=(
                ChatMessage("system", "system instructions"),
                ChatMessage("user", "solve this"),
            ),
            prompt_version="initial-v1",
            sampling=SamplingParameters(
                temperature=0.7,
                top_p=0.9,
                max_tokens=100,
                seed=2,
            ),
        )

    def test_gemini_serialization_uses_batch_api_shape(self) -> None:
        record = to_gemini_batch_record(self.request)

        self.assertEqual(record["key"], self.request.request_id)
        self.assertEqual(record["request"]["contents"][0]["role"], "user")
        self.assertEqual(
            record["request"]["generation_config"]["max_output_tokens"], 100
        )
        self.assertEqual(
            record["request"]["system_instruction"]["parts"][0]["text"],
            "system instructions",
        )

    def test_gemini_serialization_can_preserve_historical_provider_defaults(self) -> None:
        request = replace(
            self.request,
            sampling=SamplingParameters(use_provider_defaults=True),
        )

        record = to_gemini_batch_record(request)

        self.assertNotIn("generation_config", record["request"])

    def test_gemini_result_is_normalized(self) -> None:
        provider_record = {
            "key": self.request.request_id,
            "response": {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "answer"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 5,
                },
            },
        }

        result = parse_gemini_batch_record(
            provider_record,
            self.request,
            model="gemini-model",
        )

        self.assertEqual(result.raw_output, "answer")
        self.assertEqual(result.finish_reason, "STOP")
        self.assertEqual(result.usage.total_tokens, 16)
        self.assertEqual(result.prompt_version, self.request.prompt_version)
        self.assertEqual(result.sampling, self.request.sampling)

    def test_gemini_batch_status_keeps_client_alive_until_request_finishes(self) -> None:
        client = Mock()
        batch = Mock(name="batch")
        client.batches.get.return_value = batch

        with patch(
            "explain_then_adapt.data_generation.backends.gemini.create_client",
            return_value=client,
        ):
            result = get_batch("batches/test")

        self.assertIs(result, batch)
        client.batches.get.assert_called_once_with(name="batches/test")
        client.close.assert_called_once_with()

    def test_vllm_conversion_has_no_runtime_dependency(self) -> None:
        messages = to_vllm_messages(self.request)

        self.assertEqual(messages[0], {"role": "system", "content": "system instructions"})


if __name__ == "__main__":
    unittest.main()
