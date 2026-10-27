"""Retrieval-augmented question answering over EU road transport legislation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eu-transport-rag")
except PackageNotFoundError:  # running from source without installing the project
    __version__ = "0+unknown"
