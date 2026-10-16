class BiomassError(Exception):
    """Base class for package errors."""


class AuthError(BiomassError):
    """The token file is missing or the IAM refused to issue an access token."""


class DownloadError(BiomassError):
    """A file could not be downloaded after all retries."""


class QueryError(BiomassError, ValueError):
    """Invalid search parameters."""
