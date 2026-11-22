import requests

_HEAD_TIMEOUT_SECONDS = 30


def content_length(url: str) -> int | None:
    """*url*'s size from a HEAD request that follows redirects; None when the server
    doesn't say. Raises RuntimeError when the URL can't be reached."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=_HEAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to reach {url}: {e}") from e
    length = response.headers.get("content-length", "")
    return int(length) if length.isdigit() and int(length) > 0 else None
