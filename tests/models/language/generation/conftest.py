# conftest.py
import pytest


def pytest_addoption(parser):
    """
    Adds custom command-line options for pytest.
    """
    parser.addoption(
        "--model_card_path",
        action="store",
        default=None,  # Set a default or make it required based on your needs
        help="Path to the model card YAML file.",
    )


@pytest.fixture(scope="session")
def model_card_path(request):
    """
    Fixture to provide the --model_card_path option value to tests.
    """
    path = request.config.getoption("--model_card_path")
    if path is None:
        pytest.fail("The --model_card_path argument is required for this test.")
    return path
