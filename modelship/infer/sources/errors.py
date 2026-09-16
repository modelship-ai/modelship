class ModelDownloadError(Exception):
    """A validated source failed to download; `ModelDeployment` treats it as
    transient, unlike the check errors."""


class ModelSourceError(Exception):
    """A source's files on disk are wrong in a way a retry can't fix; left
    unwrapped by `ensure_downloaded`, so `ModelDeployment` reports it as fatal."""
