from datetime import UTC, datetime

from riftwatch.stats import extract


def participant(pid, team, pos, champ, win, **extra):
    base = {"participantId": pid, "puuid": f"p{pid}", "teamId": team, "teamPosition": pos,
            "championId": champ, "win": win, "kills": 2, "deaths": 3, "assists": 4,
            "riotIdGameName": f"Player{pid}", "riotIdTagline": "NA1",
            "totalMinionsKilled": 150, "neutralMinionsKilled": 10, "goldEarned": 9000,
            "totalDamageDealtToChampions": 12000, "totalDamageTaken": 15000, "visionScore": 20,
            "wardsPlaced": 8, "wardsKilled": 2, "visionWardsBoughtInGame": 3,
            "largestMultiKill": 1, "item0": 3031, "item1": 0, "item6": 3340,
            "summoner1Id": 4, "summoner2Id": 14, "champLevel": 15,
            "perks": {"statPerks": {"offense": 5008, "flex": 5008, "defense": 5011},
                      "styles": [{"style": 8000, "selections": [{"perk": 8010}, {"perk": 9111},
                                                                  {"perk": 9104}, {"perk": 8299}]},
                                 {"style": 8400, "selections": [{"perk": 8444}, {"perk": 8242}]}]}}
    base.update(extra)
    return base


def match():
    parts = [participant(1, 100, "TOP", 266, True, kills=9, largestMultiKill=3, tripleKills=1,
                         pentaKills=0, firstBloodKill=True),
             participant(2, 100, "MIDDLE", 103, True),
             participant(6, 200, "TOP", 86, False),
             participant(7, 200, "MIDDLE", 157, False)]
    return {"metadata": {"matchId": "NA1_42"},
            "info": {"queueId": 420, "gameVersion": "16.19.712.1", "gameStartTimestamp": 1_760_000_000_000,
                     "gameEndTimestamp": 1_760_001_800_000, "gameDuration": 1800,
                     "participants": parts,
                     "teams": [{"teamId": 100, "bans": [{"championId": 1, "pickTurn": 1},
                                                        {"championId": -1, "pickTurn": 2}]},
                               {"teamId": 200, "bans": [{"championId": 12, "pickTurn": 6}]}]}}


def test_participant_rows_carry_build_runes_and_lane_opponent():
    rows = {r[1]: dict(zip(extract._COLUMNS, r, strict=True)) for r in extract.participant_rows(match())}
    top = rows[1]
    assert top["opponent_champion_id"] == 86 and rows[7]["opponent_champion_id"] == 103
    assert top["team_kills"] == 11 and top["cs"] == 160 and top["patch"] == "16.19"
    assert top["items"] == [3031, 0, 0, 0, 0, 0, 3340] and top["spells"] == [4, 14]
    assert (top["keystone"], top["primary_style"], top["sub_style"]) == (8010, 8000, 8400)
    assert top["perks"] == [8010, 9111, 9104, 8299, 8444, 8242, 5008, 5008, 5011]
    assert top["largest_multikill"] == 3 and top["first_blood"] and top["win"]
    assert top["game_start"] == datetime.fromtimestamp(1_760_000_000, tz=UTC) and top["duration_s"] == 1800
    assert extract.ban_rows(match()) == [("NA1_42", 100, 1, 1), ("NA1_42", 200, 6, 12)]


def test_timeline_gives_skill_order_and_purchases_without_undone_ones():
    events = [{"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 1055, "timestamp": 3000},
              {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 2003, "timestamp": 4000},
              {"type": "SKILL_LEVEL_UP", "participantId": 1, "skillSlot": 1, "levelUpType": "NORMAL", "timestamp": 70000},
              {"type": "SKILL_LEVEL_UP", "participantId": 1, "skillSlot": 3, "levelUpType": "NORMAL", "timestamp": 130000},
              {"type": "SKILL_LEVEL_UP", "participantId": 1, "skillSlot": 4, "levelUpType": "EVOLVE", "timestamp": 140000},
              {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 3006, "timestamp": 400000},
              {"type": "ITEM_UNDO", "participantId": 1, "beforeId": 3006, "afterId": 0, "timestamp": 401000},
              {"type": "ITEM_PURCHASED", "participantId": 1, "itemId": 3047, "timestamp": 402000},
              {"type": "SKILL_LEVEL_UP", "participantId": 2, "skillSlot": 2, "timestamp": 90000}]
    out = extract.timeline_rows(events)
    assert out[1] == ("QE", [1055, 2003, 3047], [3, 4, 402])
    assert out[2] == ("W", [], [])


def test_scores_pick_the_best_player_on_each_team():
    from riftwatch.web.views import _scores

    def p(pid, team, win, k, d, a, dmg):
        return {"participant_id": pid, "team_id": team, "win": win, "kills": k, "deaths": d,
                "assists": a, "team_kills": 20, "damage": dmg, "gold": 10000, "cs": 200,
                "vision_score": 20, "duration_s": 1800}

    players = [p(1, 100, True, 12, 1, 8, 30000), p(2, 100, True, 2, 6, 3, 9000),
               p(6, 200, False, 6, 4, 5, 22000), p(7, 200, False, 0, 9, 1, 5000)]
    s = _scores(players)
    assert max(s, key=s.get) == 1 and s[1] == 10.0 and min(s, key=s.get) == 7
    assert s[6] > s[7] and all(3 <= v <= 10 for v in s.values())


def test_game_data_classifies_items_and_resolves_names():
    from riftwatch.web.gamedata import Champion, GameData, Item

    gd = GameData()
    gd.loaded = True
    gd.champions = {62: Champion(62, "MonkeyKing", "Wukong")}
    gd.by_slug = {"monkeyking": gd.champions[62], "wukong": gd.champions[62]}
    gd.items = {3031: Item(3031, "Infinity Edge", "", 3450, ["Damage"], [], 3),
                3006: Item(3006, "Berserker's Greaves", "", 1100, ["Boots"], [], 2),
                1055: Item(1055, "Doran's Blade", "", 450, ["Damage", "Lane"], [], 1)}
    assert gd.champion_by_slug("Wukong").key == "MonkeyKing"
    assert gd.champion_icon(62).endswith("/img/champion/MonkeyKing.png")
    assert gd.champion(999).name == "Champion 999" and gd.champion_icon(999) == ""
    assert gd.item(3031).is_legendary and gd.item(3006).is_boots and gd.item(1055).is_starter
    assert not gd.item(3006).is_legendary and gd.item_icon(0) == ""
