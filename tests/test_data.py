"""Universe reconstruction, Wikipedia parsing, split repair, holdout guard."""
import numpy as np
import pandas as pd
import pytest

from src.data import panels as panels_mod
from src.data.clean import repair_split_artifacts
from src.data.panels import HOLDOUT_CONFIRM, holdout_start, load_panels, save_panels
from src.data.universe import (bridge_gap, build_intervals, membership_matrix, parse_changes, parse_constituents,
                               parse_wikipedia_tables, snapshots_to_changes)

HTML = """
<table id="constituents"><thead><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th></tr></thead>
<tbody><tr><td>AAA</td><td>Alpha</td><td>Tech</td><td>2015-06-01</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>1990-01-01</td></tr>
<tr><td>META</td><td>Meta</td><td>Tech</td><td>2013-12-23</td></tr></tbody></table>
<table id="changes"><thead>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th><th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>
<tbody><tr><td>June 1, 2015</td><td>AAA</td><td>Alpha</td><td>ZZZ</td><td>Zeta</td><td>x</td></tr>
<tr><td>December 23, 2013</td><td>FB</td><td>Facebook</td><td></td><td></td><td>x</td></tr></tbody></table>
"""


def test_parse_wikipedia_tables():
    constituents, changes = parse_wikipedia_tables(HTML, aliases={"FB": "META"})
    assert constituents["ticker"].tolist() == ["AAA", "BRK-B", "META"]
    assert set(map(tuple, changes[["ticker", "action"]].to_numpy())) == {("AAA", "added"), ("ZZZ", "removed"), ("META", "added")}
    assert changes["date"].min() == pd.Timestamp("2013-12-23")


def test_membership_walk_back():
    constituents, changes = parse_wikipedia_tables(HTML, aliases={"FB": "META"})
    intervals, issues = build_intervals(constituents["ticker"].tolist(), changes, pd.Timestamp("2010-01-01"))
    assert issues == []
    dates = pd.to_datetime(["2012-01-03", "2014-01-02", "2015-05-29", "2015-06-01", "2020-01-02"])
    m = membership_matrix(intervals, dates, ["AAA", "BRK-B", "META", "ZZZ"])
    assert m["AAA"].tolist() == [False, False, False, True, True]     # effective on the change date
    assert m["ZZZ"].tolist() == [True, True, True, False, False]
    assert m["META"].tolist() == [False, True, True, True, True]
    assert m["BRK-B"].all()


def test_membership_inconsistency_is_logged_not_hidden():
    constituents, changes = parse_wikipedia_tables(HTML)              # no alias: FB is unknown today
    _, issues = build_intervals(constituents["ticker"].tolist(), changes, pd.Timestamp("2010-01-01"))
    assert [i["ticker"] for i in issues] == ["FB"]


def test_split_artifact_is_repaired_and_real_crash_is_kept():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.DataFrame({"SPLIT": [100.0] * 5 + [50.0] * 5, "CRASH": [100.0] * 5 + [42.0] * 5, "OK": [10.0] * 10}, index=idx)
    adj, _, flagged, log = repair_split_artifacts(px, px.copy(), max_abs_return=0.4, tolerance=0.02)
    assert np.allclose(adj["SPLIT"], 50.0)
    assert adj["CRASH"].tolist() == px["CRASH"].tolist()
    assert sorted(log["ticker"]) == ["CRASH", "SPLIT"]
    assert flagged["CRASH"].sum() == 1 and flagged["OK"].sum() == 0


def test_holdout_is_locked_and_every_access_is_logged(panels, tmp_path, monkeypatch):
    cfg = {"paths": {"data_dir": "d", "reports_dir": "r"}, "split": {"holdout_months": 6}}
    monkeypatch.setattr("src.config.ROOT", tmp_path)
    save_panels(panels, cfg)
    start = holdout_start(panels.dates, 6)

    research = load_panels(cfg)
    assert research.dates[-1] < start
    assert research.market_ret.index[-1] < start and research.tradable.index[-1] < start
    assert panels_mod.holdout_access_count(cfg) == 0

    with pytest.raises(PermissionError):
        load_panels(cfg, include_holdout=True)
    with pytest.raises(PermissionError):
        load_panels(cfg, include_holdout=True, confirm="yes")
    full = load_panels(cfg, include_holdout=True, confirm=HOLDOUT_CONFIRM)
    assert full.dates[-1] == panels.dates[-1]
    assert panels_mod.holdout_access_count(cfg) == 1


def test_page_without_change_log_and_gap_bridge():
    """The live article lost its change log. Membership after the old snapshot is
    bridged from 'Date added'; removals get the last addition date and are flagged."""
    no_log = HTML.split('<table id="changes">')[0]
    assert parse_changes(no_log) is None
    snapshot = parse_constituents(no_log)
    current = pd.DataFrame({"ticker": ["AAA", "META", "NEW1", "NEW2"],
                            "date_added": pd.to_datetime(["2015-06-01", "2013-12-23", "2026-03-02", "2026-05-11"])})
    gap = bridge_gap(current, snapshot, pd.Timestamp("2026-01-15", tz="UTC"))
    assert gap["approximated"].all()
    got = {(r.ticker, r.action, str(r.date.date())) for r in gap.itertuples()}
    assert got == {("NEW1", "added", "2026-03-02"), ("NEW2", "added", "2026-05-11"), ("BRK-B", "removed", "2026-05-11")}


def test_snapshots_to_changes_round_trip():
    """Member lists -> events -> walk-back must reproduce the member list on every date."""
    history = pd.DataFrame({"date": pd.to_datetime(["2010-01-04", "2012-03-01", "2015-07-01"]),
                            "tickers": ["A,B,C", "A,C,D", "C,D,BRK.B"]})
    changes, last = snapshots_to_changes(history)
    assert last["ticker"].tolist() == ["BRK-B", "C", "D"]
    intervals, issues = build_intervals(last["ticker"].tolist(), changes, pd.Timestamp("2010-01-01"))
    assert issues == []
    dates = pd.to_datetime(["2010-06-01", "2012-03-01", "2014-01-02", "2015-07-01"])
    m = membership_matrix(intervals, dates, ["A", "B", "C", "D", "BRK-B"])
    assert [sorted(m.columns[row]) for row in m.to_numpy()] == [["A", "B", "C"], ["A", "C", "D"], ["A", "C", "D"], ["BRK-B", "C", "D"]]
