"""Provider-neutral text-to-speech contracts for audio probes."""

from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SynthesisRequest:
    """Portable synthesis inputs understood by audio probes."""

    text: str
    language: str | None = None
    voice: str | None = None
    style: str | None = None
    seed: int | None = None
    sample_rate: int | None = None


@dataclass(frozen=True)
class SynthesisResult:
    """Synthesized waveform and effective provider metadata."""

    audio: object
    sample_rate: int
    provider: str
    model: str
    revision: str | None = None
    effective_options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SynthesisCapabilities:
    """Optional controls supported by a synthesis provider."""

    languages: tuple[str, ...] = ()
    voices: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()
    supports_seed: bool = False


@runtime_checkable
class SynthesisProvider(Protocol):
    """Minimal provider interface used by audio probes."""

    def capabilities(self) -> SynthesisCapabilities:
        """Return supported optional controls."""

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize one request."""


def validate_synthesis_request(
    request: SynthesisRequest, capabilities: SynthesisCapabilities
) -> None:
    """Reject requested controls that a provider does not advertise."""

    checks = (
        ("language", request.language, capabilities.languages),
        ("voice", request.voice, capabilities.voices),
        ("style", request.style, capabilities.styles),
    )
    for label, value, supported in checks:
        if value is not None and value not in supported:
            raise ValueError(f"synthesis provider does not support {label} {value!r}")
    if request.seed is not None and not capabilities.supports_seed:
        raise ValueError("synthesis provider does not support deterministic seeds")


def synthesis_identity(request: SynthesisRequest, result: SynthesisResult) -> dict:
    """Return cache/provenance identity for a completed synthesis."""

    return {
        "request": asdict(request),
        "provider": result.provider,
        "model": result.model,
        "revision": result.revision,
        "sample_rate": result.sample_rate,
        "effective_options": result.effective_options,
    }


class TransformersSynthesisProvider:
    """Lazy adapter for a Transformers text-to-audio pipeline."""

    def __init__(
        self, model: str, revision: str | None = None, voices: tuple[str, ...] = ()
    ):
        self.model = model
        self.revision = revision
        self.voices = tuple(voices)
        self._pipeline = None

    def capabilities(self) -> SynthesisCapabilities:
        """Return supported controls; advertise configured voice presets."""

        return SynthesisCapabilities(voices=self.voices)

    def _load_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise ModuleNotFoundError(
                    "Transformers synthesis requires the transformers package"
                ) from exc
            arguments = {"model": self.model}
            if self.revision:
                arguments["revision"] = self.revision
            try:
                import torch

                if torch.cuda.is_available():
                    arguments["device"] = 0
            except (ImportError, RuntimeError):
                pass
            self._pipeline = pipeline("text-to-audio", **arguments)
        return self._pipeline

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        """Synthesize plain text using a lazily loaded pipeline."""

        validate_synthesis_request(request, self.capabilities())
        pipeline = self._load_pipeline()
        # Forward a requested voice preset to the model (e.g. Bark history_prompt).
        # If the backend ignores it, distinct voices produce identical audio hashes
        # -- a signal to verify voice control actually took effect.
        call_kwargs = {}
        effective_options = {}
        if request.voice is not None:
            call_kwargs["forward_params"] = {"history_prompt": request.voice}
            effective_options["voice"] = request.voice
        output = pipeline(request.text, **call_kwargs)
        audio = output["audio"] if isinstance(output, dict) else output
        sample_rate = (
            output.get("sampling_rate", request.sample_rate)
            if isinstance(output, dict)
            else request.sample_rate
        )
        if sample_rate is None:
            raise ValueError("synthesis provider did not return a sample rate")
        return SynthesisResult(
            audio=audio,
            sample_rate=int(sample_rate),
            provider="transformers.text-to-audio",
            model=self.model,
            revision=self.revision,
            effective_options=effective_options,
        )
