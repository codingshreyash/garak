"""Load validated, user-overridable source data for audio probes."""

from dataclasses import dataclass
import json

from garak.data import path as data_path
from garak.exception import GarakException


def load_audio_source(default_filename: str) -> dict:
    """Load an audio source-data object using garak data-path precedence."""

    try:
        source_path = data_path / "audio" / default_filename
        with source_path.open("r", encoding="utf-8") as source_file:
            source = json.load(source_file)
    except (FileNotFoundError, GarakException) as exc:
        raise ValueError(
            f"audio source data not found: audio/{default_filename}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in audio source data: {source_path}") from exc
    except OSError as exc:
        raise ValueError(f"unable to read audio source data: {source_path}") from exc

    if not isinstance(source, dict):
        raise ValueError(f"audio source data must be an object: {source_path}")
    return source


def required_text(record: dict, key: str, context: str) -> str:
    """Return a required non-empty text field from a source-data record."""

    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty {key}")
    return value


def required_records(source: dict, key: str, context: str) -> list[dict]:
    """Return a required non-empty list of object records."""

    records = source.get(key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{context} requires a non-empty {key} list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{context} {key} entries must be objects")
    return records


def text_groups(value, context: str) -> tuple[tuple[str, ...], ...]:
    """Validate alternative-term groups used by reliability detectors."""

    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of string lists")
    groups = []
    for group in value:
        if (
            not isinstance(group, list)
            or not group
            or any(not isinstance(term, str) or not term for term in group)
        ):
            raise ValueError(f"{context} must contain non-empty string lists")
        groups.append(tuple(group))
    return tuple(groups)


@dataclass(frozen=True)
class ToolRiskControl:
    """Benign matched control for one tool-risk source case."""

    case_id: str
    source_text: str
    required_response_terms: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ToolRiskCase:
    """Tool-risk source request and optional matched-control metadata."""

    case_id: str
    intent: str
    source_text: str
    action_signature: str
    harmful_source_text: str | None = None
    benign_control: ToolRiskControl | None = None


@dataclass(frozen=True)
class ToolRiskSource:
    """Validated tool-risk cases and semantic framing templates."""

    cases: tuple[ToolRiskCase, ...]
    semantic_strategies: dict[str, str]


def load_tool_risk_source() -> ToolRiskSource:
    """Load tool-risk cases through garak data-path precedence."""

    source = load_audio_source("tool_risk.json")
    cases = []
    seen_ids = set()
    for index, record in enumerate(
        required_records(source, "cases", "tool-risk source data")
    ):
        context = f"tool-risk case {index}"
        case_id = required_text(record, "case_id", context)
        if case_id in seen_ids:
            raise ValueError(f"duplicate tool-risk case_id: {case_id}")
        seen_ids.add(case_id)

        paired = record.get("paired")
        harmful_source_text = None
        benign_control = None
        if paired is not None:
            if not isinstance(paired, dict):
                raise ValueError(f"{context} paired metadata must be an object")
            harmful_source_text = required_text(
                paired, "harmful_source_text", f"{context} paired metadata"
            )
            benign = paired.get("benign_control")
            if not isinstance(benign, dict):
                raise ValueError(f"{context} requires a benign_control object")
            benign_control = ToolRiskControl(
                case_id=required_text(benign, "case_id", f"{context} benign control"),
                source_text=required_text(
                    benign, "source_text", f"{context} benign control"
                ),
                required_response_terms=text_groups(
                    benign.get("required_response_terms"),
                    f"{context} benign required_response_terms",
                ),
            )

        cases.append(
            ToolRiskCase(
                case_id=case_id,
                intent=required_text(record, "intent", context),
                source_text=required_text(record, "source_text", context),
                action_signature=required_text(record, "action_signature", context),
                harmful_source_text=harmful_source_text,
                benign_control=benign_control,
            )
        )

    strategies = source.get("semantic_strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("tool-risk source data requires semantic_strategies")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(template, str)
        or "{source_text}" not in template
        for name, template in strategies.items()
    ):
        raise ValueError(
            "tool-risk semantic strategies require named templates containing "
            "{source_text}"
        )
    for name, template in strategies.items():
        try:
            template.format(source_text="source request")
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                f"invalid tool-risk semantic strategy template: {name}"
            ) from exc
    return ToolRiskSource(tuple(cases), dict(strategies))


@dataclass(frozen=True)
class InstructionPrioritySource:
    """Default content and conditions for instruction-priority trials."""

    conditions: tuple[str, ...]
    foreground_phrase: str
    secondary_phrase: str
    neutral_audio_text: str


def load_instruction_priority_source() -> InstructionPrioritySource:
    """Load instruction-priority content through garak data-path precedence."""

    source = load_audio_source("instruction_priority.json")
    conditions = source.get("conditions")
    if (
        not isinstance(conditions, list)
        or not conditions
        or any(
            not isinstance(condition, str) or not condition for condition in conditions
        )
    ):
        raise ValueError("instruction-priority conditions must be non-empty strings")
    return InstructionPrioritySource(
        conditions=tuple(conditions),
        foreground_phrase=required_text(
            source, "foreground_phrase", "instruction-priority source data"
        ),
        secondary_phrase=required_text(
            source, "secondary_phrase", "instruction-priority source data"
        ),
        neutral_audio_text=required_text(
            source, "neutral_audio_text", "instruction-priority source data"
        ),
    )
