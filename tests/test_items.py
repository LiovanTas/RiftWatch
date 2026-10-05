from riftwatch.features.extract import extract
from riftwatch.features.items import ItemRules, build_timings, purchases
from tests.fixtures import build_game

DATA = {
    "1001": {"gold": {"total": 300, "purchasable": True}, "maps": {"11": True}, "tags": ["Boots"], "into": ["3006"]},
    "3006": {"gold": {"total": 1100, "purchasable": True}, "maps": {"11": True}, "tags": ["Boots"]},
    "1036": {"gold": {"total": 350, "purchasable": True}, "maps": {"11": True}, "tags": ["Damage"], "into": ["3031"]},
    "3031": {"gold": {"total": 3400, "purchasable": True}, "maps": {"11": True}, "tags": ["Damage", "CriticalStrike"]},
    "3072": {"gold": {"total": 3200, "purchasable": True}, "maps": {"11": True}, "tags": ["Damage"]},
    "2003": {"gold": {"total": 50, "purchasable": True}, "maps": {"11": True}, "tags": ["Consumable"]},
    "9999": {"gold": {"total": 3000, "purchasable": True}, "maps": {"30": True}, "tags": ["Damage"]},   # Arena only
    "3340": {"gold": {"total": 0, "purchasable": True}, "maps": {"11": True}, "tags": ["Trinket"]},
}
RULES = ItemRules.from_ddragon(DATA)


def buy(ts, pid, item):
    return {"type": "ITEM_PURCHASED", "timestamp": ts, "participantId": pid, "itemId": item}


def test_rules():
    assert RULES.legendary == {3031, 3072} and RULES.boots == {3006}


def test_undo_cancels_the_last_matching_purchase():
    events = [buy(1000, 1, 3031),
              {"type": "ITEM_UNDO", "timestamp": 3000, "participantId": 1, "beforeId": 3031, "afterId": 0},
              buy(600_000, 1, 3031)]
    assert purchases(events, 1) == [(600_000, 3031)]


def test_build_timings():
    events = [buy(90_000, 1, 1001), buy(400_000, 1, 3006), buy(640_000, 1, 3031),
              buy(900_000, 1, 3031), buy(1_000_000, 1, 3072), buy(500_000, 2, 3072)]
    t = build_timings(events, 1, RULES)
    assert t == {"boots_min": 6.67, "first_item_min": 10.67, "second_item_min": 16.67}


def test_extract_adds_build_timings_only_with_rules():
    match, timeline = build_game(events=[buy(640_000, 4, 3031)])
    assert "first_item_min" not in extract(match, timeline).participants[4].metrics
    assert extract(match, timeline, items=RULES).participants[4].metrics["first_item_min"] == 10.67
