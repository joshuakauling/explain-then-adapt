"""Pure prompt builders for initial generation, judging, and trace rewriting."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from explain_then_adapt.arc.augmented_keys import parse_order_mapping
from explain_then_adapt.arc.formatting import format_puzzle_to_string
from explain_then_adapt.arc.types import Example

from .hints import Hint
from .records import AugmentationSpec, ChatMessage


INITIAL_PROMPT_VERSION = "initial-v1"
JUDGE_PROMPT_VERSION = "judge-v1"
REWRITE_PROMPT_VERSION = "rewrite-v1"

INITIAL_SYSTEM_PROMPT = (
    "You are a logic-puzzle specialist. Analyze grid-based puzzles and produce "
    "a detailed reasoning trace that explains the transformation from input to output."
)

INITIAL_RULES = """Important rules:
- The output MUST start with <think> as the first characters.
- The output MUST contain exactly one <think> ... </think> block.
- After </think>, output exactly these sections in this order:
  1) "General natural language description:"
  2) "General steps:"
- Do not add text outside the specified format.
- If uncertain, retain the full format and use "UNKNOWN" for unknown content.
- Write the entire output in English.
- Produce one reasoning block covering all provided examples together.
- Use 1-based row and column indices, from top to bottom and left to right.
- Analyze all inputs in section 1 and all outputs in section 2; do not interleave them.
- Prefer structure-level descriptions such as clusters, windows, masks, quadrants,
  bounding boxes, separators, stacks, and downsampling over per-cell coordinates.
- State relevant domain choices such as connectivity, grouping factors, and tie-breaks.
- If multiple rules fit, choose the simplest rule consistent with every example."""

INITIAL_FORMAT = """Output format (strict):
<think>
1) INPUT ANALYSIS
- Describe the input grid sizes, values, background, and structures for every example.
- Use coordinates only when essential.

2) OUTPUT ANALYSIS
- Describe the output grid sizes, values, background, and structures for every example.
- Briefly compare overall output structure and size with the inputs, but leave the
  detailed mapping to section 3.

3) TRANSFORMATION ANALYSIS
- Explore plausible hypotheses by comparing inputs and outputs.
- Verify the selected hypothesis against every demonstration.
- State one concrete, example-independent rule covering geometry, values,
  parameters, conventions, and deterministic tie handling where relevant.

4) STEPS FOR THE TRANSFORMATION
1. Determine the output grid size.
2. Identify the relevant structures.
3. Derive the required parameters.
4. Apply the mapping.
5. Post-process and return the result.
</think>

General natural language description:
A concise, puzzle-independent description of the transformation.

General steps:
A concise numbered procedure for applying the transformation."""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict ARC reasoning auditor. Return one JSON object and no other text."
)

REWRITE_SYSTEM_PROMPT = (
    "You adapt an existing ARC reasoning trace to geometrically transformed, "
    "value-remapped, and reordered demonstrations."
)

STYLE_MODES: Dict[str, str] = {
    "neutral": "Write clearly and factually without stylistic commentary.",
    "reflective": (
        "Write as if thinking aloud, briefly questioning and checking observations "
        "before confirming them."
    ),
    "concise": "Use short, efficient sentences focused on essential reasoning.",
    "analytical": (
        "Make comparisons, invariants, and cross-example consistency checks explicit."
    ),
    "elaborate": (
        "Explain reasoning steps thoroughly with explicit causal connections."
    ),
}

GEOMETRY_DESCRIPTIONS: Dict[str, str] = {
    "ID": "The grid is unchanged. Dimensions HxW -> HxW; directions are unchanged.",
    "R90": (
        "Rotate 90 degrees clockwise. Dimensions HxW -> WxH; top -> right, "
        "right -> bottom, bottom -> left, left -> top."
    ),
    "R180": (
        "Rotate 180 degrees. Dimensions HxW -> HxW; top and bottom swap, "
        "and left and right swap."
    ),
    "R270": (
        "Rotate 270 degrees clockwise. Dimensions HxW -> WxH; top -> left, "
        "right -> top, bottom -> right, left -> bottom."
    ),
    "FH": (
        "Reflect across the horizontal axis. Dimensions HxW -> HxW; "
        "top and bottom swap."
    ),
    "FV": (
        "Reflect across the vertical axis. Dimensions HxW -> HxW; "
        "left and right swap."
    ),
    "FD1": (
        "Reflect across the main diagonal. Dimensions HxW -> WxH; "
        "top -> left and left -> top."
    ),
    "FD2": (
        "Reflect across the anti-diagonal. Dimensions HxW -> WxH; "
        "top -> right, right -> top, bottom -> left, left -> bottom."
    ),
}


@dataclass(frozen=True)
class FewShotExample:
    """One fully worked demonstration embedded in an initial prompt."""

    task_id: str
    puzzle: List[Example]
    trace: str
    hint: Optional[Hint] = None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.trace.strip():
            raise ValueError("few-shot task_id and trace must not be empty.")


def _format_hint_section(hint: Optional[Hint]) -> str:
    if hint is None:
        return ""
    return "## Hints:\n" + hint.format()


def _format_few_shot(example: FewShotExample) -> str:
    parts = [
        f"### Few-shot task: {example.task_id}",
        "## Puzzle:\n" + format_puzzle_to_string(example.puzzle, delimiter=""),
    ]
    if example.hint is not None:
        parts.append(_format_hint_section(example.hint))
    parts.append("## Reasoning trace:\n" + example.trace.strip())
    return "\n\n".join(parts)


def build_initial_prompt(
    puzzle: List[Example],
    *,
    hint: Optional[Hint],
    few_shots: Tuple[FewShotExample, ...],
) -> str:
    """Build the user prompt for one initial reasoning-trace candidate."""
    input_description = (
        "The target contains ARC input/output demonstration pairs. A complete "
        "manually curated hint is included as background."
        if hint is not None
        else "The target contains ARC input/output demonstration pairs and no task-specific hint."
    )
    hint_rule = (
        "- Do not quote, translate, or reference the hint; derive and verify the rule "
        "from the examples."
        if hint is not None
        else "- Derive and verify every claim directly from the examples."
    )
    parts = [input_description, INITIAL_RULES, hint_rule, INITIAL_FORMAT]
    if few_shots:
        parts.append(
            "## Few-shot demonstrations:\n\n"
            + "\n\n".join(_format_few_shot(example) for example in few_shots)
        )
    target_parts = [
        "# Puzzle to solve",
        "## Puzzle:\n" + format_puzzle_to_string(puzzle, delimiter=""),
    ]
    if hint is not None:
        target_parts.append(_format_hint_section(hint))
    parts.append("\n\n".join(target_parts))
    return "\n\n".join(parts).strip() + "\n"


def build_initial_messages(
    puzzle: List[Example],
    *,
    hint: Optional[Hint],
    few_shots: Tuple[FewShotExample, ...],
) -> Tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content=INITIAL_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=build_initial_prompt(puzzle, hint=hint, few_shots=few_shots),
        ),
    )


def build_judge_prompt(
    puzzle: List[Example],
    candidate_trace: str,
    *,
    hint: Optional[Hint],
) -> str:
    """Build a strict JSON-verdict prompt for one candidate trace."""
    if not candidate_trace.strip():
        raise ValueError("candidate_trace must not be empty.")
    hint_criterion = (
        "- No hint leakage: the hint is not quoted, translated, or referenced."
        if hint is not None
        else "- Evidence grounding: every claim is supported by the demonstrations."
    )
    hint_section = (
        "\n\nGerman hint (background only):\n" + hint.format()
        if hint is not None
        else ""
    )
    return f"""Audit the candidate reasoning trace against the ARC demonstrations.

Evaluation criteria:
- English-only: the trace contains no German sentences.
{hint_criterion}
- Single general rule: one puzzle-independent rule covers every example.
- Evidence completeness: sections 1 and 2 cover all examples.
- Structure-level input and output descriptions are preferred; coordinates are minimal.
- No full grid is quoted or reproduced in the trace.
- Output analysis contains only high-level comparison; detailed mapping belongs in section 3.
- Transformation analysis maps input structures to output structures and verifies the rule.
- Transformation steps are concise, implementation-ready, ordered, and consistent.
- The final description and steps introduce no claims absent from the reasoning block.

Return JSON only, with this exact top-level shape:
{{
  "verdict": "pass" | "fail",
  "criteria": {{
    "ENGLISH_ONLY": true | false,
    "EVIDENCE_GROUNDED": true | false,
    "SINGLE_RULE": true | false,
    "EVIDENCE_COMPLETE": true | false,
    "STRUCTURE_LEVEL_INPUT": true | false,
    "STRUCTURE_LEVEL_OUTPUT": true | false,
    "NO_FULL_GRID_QUOTE": true | false,
    "OUTPUT_NO_EARLY_MAPPING": true | false
  }},
  "violations": [{{"code": "...", "where": "short snippet"}}],
  "summary": "one short sentence"
}}

Puzzle:
{format_puzzle_to_string(puzzle, delimiter="")}{hint_section}

Candidate reasoning trace:
{candidate_trace.strip()}
"""


def build_judge_messages(
    puzzle: List[Example],
    candidate_trace: str,
    *,
    hint: Optional[Hint],
) -> Tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=build_judge_prompt(puzzle, candidate_trace, hint=hint),
        ),
    )


def describe_augmentation(spec: AugmentationSpec) -> str:
    """Describe structured augmentation fields for the rewrite model."""
    value_mapping = ", ".join(
        f"{source} -> {target}" for source, target in enumerate(spec.value_mapping)
    )
    order_mapping = ", ".join(
        f"new example {new + 1} = old example {old + 1}"
        for new, old in enumerate(parse_order_mapping(spec.order_mapping))
    )
    return "\n".join(
        (
            "This transformation affects geometry, cell values, and example order.",
            "Geometric transformation: "
            + GEOMETRY_DESCRIPTIONS[spec.transformation_code],
            "Value mapping: " + value_mapping + ".",
            "Example order mapping: " + order_mapping + ".",
        )
    )


def build_rewrite_prompt(
    old_puzzle: List[Example],
    old_trace: str,
    new_puzzle: List[Example],
    spec: AugmentationSpec,
) -> str:
    """Build a prompt that adapts an accepted trace to an augmented task."""
    if not old_trace.strip():
        raise ValueError("old_trace must not be empty.")
    try:
        style_instruction = STYLE_MODES[spec.style]
    except KeyError as error:
        raise ValueError(
            f"unknown rewrite style {spec.style!r}; expected one of {tuple(STYLE_MODES)}."
        ) from error

    return f"""Adapt the existing reasoning trace to the transformed puzzle.

Rules:
- Preserve the logical meaning, section structure, and reasoning flow.
- Update all directional, coordinate, value, and example-index references.
- Refer only to the current transformed grids and example indices.
- Do not mention the old puzzle, transformation specification, or adaptation process.
- Do not solve the task from scratch or introduce new assumptions.
- Keep approximately the same reasoning granularity.
- {style_instruction}
- Use 1-based row and column indices.
- Preserve every required heading and exactly one <think> ... </think> block.
- After </think>, preserve "General natural language description:" and then
  "General steps:".
- Output only the adapted reasoning trace.

Old puzzle:
{format_puzzle_to_string(old_puzzle, delimiter="")}

Old reasoning trace:
{old_trace.strip()}

Transformation specification:
{describe_augmentation(spec)}

New puzzle:
{format_puzzle_to_string(new_puzzle, delimiter="")}
"""


def build_rewrite_messages(
    old_puzzle: List[Example],
    old_trace: str,
    new_puzzle: List[Example],
    spec: AugmentationSpec,
) -> Tuple[ChatMessage, ...]:
    return (
        ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT),
        ChatMessage(
            role="user",
            content=build_rewrite_prompt(old_puzzle, old_trace, new_puzzle, spec),
        ),
    )
