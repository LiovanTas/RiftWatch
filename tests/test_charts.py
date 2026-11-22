"""The champion page's win-rate charts."""

from riftwatch.web.views import win_rate_chart


def test_win_rate_bars_show_samples_and_margins_and_flag_small_ones():
    svg = win_rate_chart([{"label": "under 20", "games": 8, "win_rate": 37.5},
                          {"label": "20-25", "games": 27, "win_rate": 74.1},
                          {"label": "25-30", "games": 300, "win_rate": 51.0}], "Win rate by game length")
    assert "74%" in svg and "300 games" in svg and "±" in svg
    assert 'class="col down few"' in svg and 'class="col up few"' in svg and 'class="col up"' in svg
    assert "Only 335 stored games" in svg                       # most buckets are small: say so
    big = win_rate_chart([{"label": "26.19", "games": 400, "win_rate": 52.0},
                          {"label": "26.22", "games": 380, "win_rate": 49.0}], "Win rate by patch")
    assert "Only" not in big and 'class="col down"' in big
    assert win_rate_chart([], "x") is None


def test_coaching_text_drops_the_comparison_bookkeeping():
    from riftwatch.report.html import _first_sentence, _plain

    t = ("From minute 15 to minute 30, wards placed stayed in the bottom quarter of comparable players "
         "(vs Platinum jungle, same role, n=526); at minute 30 it was 3 (median 6, 3 below it).")
    assert _plain(t) == ("From minute 15 to minute 30, wards placed stayed in the bottom quarter of comparable "
                         "players; at minute 30 it was 3 (median 6, 3 below it).")
    assert _plain("enemy wards cleared: 15; median 3, 12 above it. Master+ median: 4.") == \
        "Enemy wards cleared: 15; median 3, 12 above it."
    long = ("At 36:55, Smolder killed Viego (your death): your team's chance to win went from 22% to 16%, and -6.9 "
            "points of it are your Swing; the jump splits into a player down -5.8. You died at level 17.")
    assert _first_sentence(long).endswith("are your Swing.")
