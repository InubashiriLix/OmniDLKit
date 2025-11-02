#!/usr/bin/env python3
"""Helper entrypoint to run pytest with local defaults."""

import os
import sys

os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

import pytest  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(pytest.main())
