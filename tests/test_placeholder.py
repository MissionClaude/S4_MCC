"""Placeholder test to verify tooling is wired correctly."""


def test_project_imports() -> None:
    """Verify the package is importable."""
    import scenario_4_dev_productivity

    assert scenario_4_dev_productivity.__version__ == "0.1.0"
