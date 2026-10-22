"""HTTP error statuses for the ``async_http_json`` dialects.

A failed sample keeps ``str(exc)``, and ``httpx.HTTPStatusError`` omits the
provider's reason, which is in the body. The OpenAI SDK includes it; so do these.

AI-Generated Code - Claude Opus 5.5 (Anthropic)
"""

import json

import httpx

#: Longest provider message kept.
PROVIDER_ERROR_MESSAGE_LIMIT = 500


def raise_for_provider_status(response: httpx.Response, provider: str) -> None:
    """Raise httpx's status error with the provider's ``error.message`` appended.

    A non-JSON body (a proxy's) is appended as text. Read a streamed body first.
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
