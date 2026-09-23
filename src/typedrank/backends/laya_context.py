"""Version-pinned, fail-closed input guard for Laya 0.3.7 sequence construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import CapabilityError, ContextLimitError


def validate_local_context(
    router: Any, state: Any, questions: Mapping[str, Any], checkpoint: str, limit: int
) -> None:
    """Check every component before Laya can silently truncate it.

    This uses the tokenizer and configuration of the actual loaded checkpoint.
    The narrow private Agent seam is certified only for Laya 0.3.7.
    """
    try:
        from laya.common import render_options, serialize_state

        agent = router.load(checkpoint)
        tokenizer = agent.tok
        config = agent.cfg
        to_internal = agent._to_internal
        max_len = min(limit, config.get("max_len", 512))
        head_limit = config.get("head_max_len", 192)
        mask = tokenizer.mask_token
        state_ids = tokenizer(serialize_state(state).replace(mask, " "), add_special_tokens=False)[
            "input_ids"
        ]
        for question in questions.values():
            internal = to_internal(question)
            options = render_options(internal)
            option_ids = [
                tokenizer(" " + option.replace(mask, " "), add_special_tokens=False)["input_ids"]
                for option in options
            ]
            if any(len(option) > 48 for option in option_ids):
                raise ContextLimitError("Laya would truncate a decision option")
            option_total = sum(1 + len(option) for option in option_ids)
            instruction_budget = head_limit - option_total
            if instruction_budget < 16:
                raise ContextLimitError("Laya would truncate decision options")
            head_ids = tokenizer(
                f"{internal['t']} question: {str(internal['ins']).replace(mask, ' ')}",
                add_special_tokens=False,
            )["input_ids"]
            if len(head_ids) > max(8, instruction_budget):
                raise ContextLimitError("Laya would truncate decision instructions")
            available_state = max_len - (len(head_ids) + option_total + 4)
            if len(state_ids) > available_state:
                raise ContextLimitError("Laya would truncate candidate state")
    except ContextLimitError:
        raise
    except (AttributeError, ImportError, KeyError, TypeError, ValueError) as exc:
        raise CapabilityError(
            "strict Laya context validation requires the Laya 0.3.7 tokenizer layout"
        ) from exc
