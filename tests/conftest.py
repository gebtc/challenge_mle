"""Pytest configuration and shared fixtures."""

import warnings

# Suppress deprecation and future warnings from third-party libraries
# during test runs to keep output readable.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
