"""Lagoon error types."""


class LagoonError(Exception):
    """Base error for all lagoon failures."""


class LagoonVersionError(LagoonError):
    """Manifest version mismatch."""


class LagoonChecksumError(LagoonError):
    """File checksum verification failed."""
