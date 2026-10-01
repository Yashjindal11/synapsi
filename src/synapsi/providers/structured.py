from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from synapsi.core.errors import StructuredOutputError
from synapsi.providers.base import Completion, CompletionRequest, Message

T = TypeVar("T", bound=BaseModel)

CompleteFn = Callable[[CompletionRequest], Awaitable[Completion]]

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object in ``text``, tolerating code fences and prose."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    raise StructuredOutputError("no JSON object found in model output", raw=text)


async def generate_structured(
    complete: CompleteFn,
    request: CompletionRequest,
    schema: type[T],
    *,
    repair_attempts: int = 1,
) -> tuple[T, list[Completion]]:
    """Call a model and validate its output against ``schema``.

    On invalid output the model is shown the validation error and asked once
    (by default) to correct it. All completions are returned for accounting.
    """
    request = request.model_copy(
        update={"json_mode": True, "response_schema": schema.model_json_schema()}
    )
    completions: list[Completion] = []
    last_error = ""
    for attempt in range(repair_attempts + 1):
        completion = await complete(request)
        completions.append(completion)
        try:
            data = extract_json(completion.text)
            return schema.model_validate(data), completions
        except (StructuredOutputError, ValidationError) as exc:
            last_error = _short_error(exc)
            if attempt == repair_attempts:
                break
            request = request.model_copy(
                update={
                    "messages": [
                        *request.messages,
                        Message(role="assistant", content=completion.text[:4000]),
                        Message(
                            role="user",
                            content=(
                                "Your previous reply was not valid for the required JSON "
                                f"format: {last_error}\nReturn only the corrected JSON object."
                            ),
                        ),
                    ]
                }
            )
    raise StructuredOutputError(
        f"{schema.__name__}: {last_error}", raw=completions[-1].text if completions else ""
    )


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:5]]
        return "; ".join(parts)
    return str(exc)
