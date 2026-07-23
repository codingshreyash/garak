import pytest

from garak.resources.audio.synthesis import (
    SynthesisCapabilities,
    SynthesisRequest,
    SynthesisResult,
    TransformersSynthesisProvider,
    synthesis_identity,
    validate_synthesis_request,
)


def test_unsupported_provider_controls_fail_before_synthesis():
    request = SynthesisRequest(text="hello", style="angry")

    with pytest.raises(ValueError, match="does not support style"):
        validate_synthesis_request(request, SynthesisCapabilities())


def test_synthesis_identity_covers_material_cache_inputs():
    request = SynthesisRequest(
        text="hello", language="en", voice="speaker-a", seed=4, sample_rate=16000
    )
    result = SynthesisResult(
        audio=[0.0],
        sample_rate=16000,
        provider="test",
        model="tts-a",
        revision="abc123",
    )

    identity = synthesis_identity(request, result)

    assert identity["request"]["voice"] == "speaker-a", "identity retains voice"
    assert identity["revision"] == "abc123", "identity retains model revision"


def test_transformers_provider_loads_lazily_and_normalizes_output(monkeypatch):
    calls = []

    class FakePipeline:
        def __call__(self, text):
            return {"audio": [0.1], "sampling_rate": 8000}

    def fake_load():
        calls.append("loaded")
        return FakePipeline()

    provider = TransformersSynthesisProvider("test-model", revision="rev")
    monkeypatch.setattr(provider, "_load_pipeline", fake_load)

    result = provider.synthesize(SynthesisRequest(text="hello"))

    assert calls == ["loaded"], "provider loads only when synthesis is requested"
    assert result.sample_rate == 8000, "provider normalizes the effective sample rate"
    assert result.revision == "rev", "provider records the configured revision"
