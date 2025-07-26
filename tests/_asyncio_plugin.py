import asyncio
import pytest


def pytest_addoption(parser):
    import sys

    if "pytest_asyncio" in sys.modules:
        # pytest-asyncio already provides this option
        return
    parser.addoption(
        "--asyncio-mode",
        action="store",
        default="auto",
        help="Dummy asyncio mode option for tests",
    )

@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
