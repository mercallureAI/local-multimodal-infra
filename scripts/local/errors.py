"""Shared script exceptions."""


class SmokeError(Exception):
    """A real smoke failure that should produce a non-zero exit code."""
