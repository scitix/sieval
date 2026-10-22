"""HTTP error statuses for the ``async_http_json`` dialects.

``httpx.HTTPStatusError`` says only the status, the reason phrase and the URL;
the provider's own reason (a per-model quota, an unknown parameter) is in the
body. A failed sample keeps ``str(exc)``, so without the body it records that a
request failed but not why -- the OpenAI SDK puts the body in its message, so
an Anthropic or Google judge failed with less said than an OpenAI one.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json

import httpx

#: Longest provider message kept: it is the diagnostic, not a body dump.
PROVIDER_ERROR_MESSAGE_LIMIT = 500


def raise_for_provider_status(response: httpx.Response, provider: str) -> None:
    """Raise httpx's status error with the provider's ``error.message`` appended.

    A body that is not the provider's JSON error (a proxy in front of it) is
    appended as text. Read the body of a streamed response before calling.
    """

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        message = _provider_message(response)
        if not message:
            raise
        if len(message) > PROVIDER_ERROR_MESSAGE_LIMIT:
            message = message[:PROVIDER_ERROR_MESSAGE_LIMIT] + "..."
        raise httpx.HTTPStatusError(
            f"{exc}\n{provider}: {message}",
            request=exc.request,
            response=exc.response,
        ) from None


def _provider_message(response: httpx.Response) -> str:
    try:
        payload = json.loads(response.text)
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str) and message:
        return message
    return response.text.strip()
