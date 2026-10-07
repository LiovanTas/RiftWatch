import pytest


@pytest.fixture(autouse=True)
def no_network_data_dragon(monkeypatch):
    """Feature extraction looks up item rules and champion names on Data Dragon; tests never
    touch the network (item and name tests pass their own)."""
    monkeypatch.setattr("riftwatch.features.store._item_rules", lambda game_version: None)
    monkeypatch.setattr("riftwatch.features.store._champion_names", lambda game_version: None)
