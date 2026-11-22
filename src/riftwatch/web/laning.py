"""The laning phase of one game for one player (the match page's Insights tab): the first 14
minutes, until turret plates fall.

From the timeline's minute frames and events:

  * gold, CS and XP against the lane opponent at 5, 10 and 14 minutes;
  * how the minutes went: farming, away (in a play, out of lane or dead) and backs, by the same
    rules Swing's farm cost uses (winprob.farm);
  * farming pace: CS in the minutes spent farming against what's typical for the role in the
    same minutes of a game (farm.json), so a lane that was often away isn't marked down for it;
  * the farm given up for plays before 14:00, from the game's Swing (what it cost beyond a
    typical play like it is what Swing charged);
  * turret plates the player took and plates their own lane lost, kills, deaths and assists.
"""

from __future__ import annotations

from typing import Any

from riftwatch.winprob import farm as farm_calc
from riftwatch.winprob.state import Replay, compact, team_of

LANING_MINUTES = farm_calc.LANING_ENDS
CHECKS = (5, 10, 14)
LANE_OF = {"TOP": "TOP_LANE", "MIDDLE": "MID_LANE", "BOTTOM": "BOT_LANE", "UTILITY": "BOT_LANE"}


def _frame(frames: list[dict], minute: int, pid: int) -> dict | None:
    return (frames[minute].get(str(pid)) or None) if minute < len(frames) else None


def report(timeline: dict | None, me_pid: int, opp_pid: int | None, role: str | None, sw: dict | None,
           table: dict | None) -> dict[str, Any] | None:
    """The laning card's numbers, or None without a timeline or a role."""
    if not timeline or not role:
        return None
    frames, events = compact(timeline)
    if len(frames) < 6:
        return None
    replay = Replay(frames, events)
    end = min(LANING_MINUTES, len(frames) - 1)

    checks = []
    for minute in CHECKS:
        a, b = _frame(frames, minute, me_pid), _frame(frames, minute, opp_pid) if opp_pid else None
        if not a:
            continue
        cs_a = farm_calc.cs(frames[minute], me_pid)
        row = {"minute": minute, "cs": round(cs_a), "gold": a.get("totalGold", 0), "xp": a.get("xp", 0),
               "level": a.get("level", 0)}
        if b:
            row.update(cs_diff=round(cs_a - farm_calc.cs(frames[minute], opp_pid)),
                       gold_diff=a.get("totalGold", 0) - b.get("totalGold", 0), xp_diff=a.get("xp", 0) - b.get("xp", 0))
        checks.append(row)

    out: dict[str, Any] = {"checks": checks, "minutes": end, "role": farm_calc.ROLE_NAMES.get(role, role.lower())}
    if role in farm_calc.ROLES:
        table = table or farm_calc.default_table()
        plays = farm_calc.involvement(replay.events).get(me_pid, [])
        deaths = farm_calc.death_spans(replay).get(me_pid, [])
        kinds = farm_calc.minutes(frames, me_pid, role, deaths, plays)[:end]
        took = [farm_calc.cs(frames[m + 1], me_pid) - farm_calc.cs(frames[m], me_pid) for m in range(end)]
        line = table["cs_rate"][role]
        farming = [m for m in range(1, end) if kinds[m] == "farm"]
        typical = sum(float(line[min(m, len(line) - 1)]) for m in farming)
        out.update(farm_minutes=len(farming), away_minutes=sum(1 for k in kinds[1:] if k == "away"),
                   base_minutes=sum(1 for k in kinds[1:] if k == "base"),
                   pace=round(100 * sum(took[m] for m in farming) / typical) if typical > 0 and len(farming) >= 3 else None,
                   strip=[{"m": m, "kind": k, "cs": round(took[m])} for m, k in enumerate(kinds) if m >= 1])

    # Farm given up for plays before plates fall, from the game's Swing.
    given = [f for m in (sw or {}).get("moments") or [] if m["t"] < LANING_MINUTES * 60_000
             for f in m.get("farm") or [] if f["pid"] == me_pid]
    if given:
        out["given"] = {"plays": len(given), "cs": round(sum(f["cs"] for f in given), 1),
                        "gold": round(sum(f["gold"] for f in given)), "xp": round(sum(f["xp"] for f in given)),
                        "worth": round(sum(f.get("worth", f["cost"]) for f in given), 1),
                        "charged": round(sum(f["cost"] for f in given), 1)}

    lane = LANE_OF.get(role)
    mine = team_of(me_pid)
    plates_took = plates_lost = kills = deaths_n = assists = 0
    for e in replay.events:
        if e.get("timestamp", 0) >= LANING_MINUTES * 60_000:
            break
        if e.get("type") == "TURRET_PLATE_DESTROYED":
            if e.get("killerId") == me_pid:
                plates_took += 1
            if lane and e.get("laneType") == lane and e.get("teamId") == mine:
                plates_lost += 1
        elif e.get("type") == "CHAMPION_KILL":
            kills += e.get("killerId") == me_pid
            deaths_n += e.get("victimId") == me_pid
            assists += me_pid in (e.get("assistingParticipantIds") or [])
    out.update(plates_took=plates_took, plates_lost=plates_lost if lane else None,
               kills=kills, deaths=deaths_n, assists=assists)
    last = checks[-1] if checks else None
    out["verdict"] = _verdict(last, out.get("pace"))
    return out


def _verdict(last: dict | None, pace: int | None) -> str | None:
    """One line on how the lane went, from the last check against the lane opponent."""
    if not last or "gold_diff" not in last:
        return None
    g = last["gold_diff"]
    where = ("won lane" if g >= 600 else "slightly ahead" if g >= 200 else "even lane" if g > -200
             else "slightly behind" if g > -600 else "lost lane")
    return f"{where.capitalize()} at {last['minute']}:00 ({g:+,} gold)" + (f", farming at {pace}% of typical pace" if pace else "")
