import pytest


@pytest.fixture(autouse=True)
def no_network_item_rules(monkeypatch):
    """Feature extraction looks up item rules on Data Dragon; tests never touch the network
    (item tests pass their own rules)."""
    monkeypatch.setattr("riftwatch.features.store._item_rules", lambda game_version: None)
