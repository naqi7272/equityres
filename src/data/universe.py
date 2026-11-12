"""Approximate point-in-time S&P 500 membership from Wikipedia.

Method: start from today's constituents and walk the "Selected changes" table
backwards in time. Going back past a change date, a ticker that was added on
that date leaves the set, and a ticker that was removed re-enters it.

Known limits (reported, not hidden): the changes table is hand-maintained and
incomplete, tickers get renamed and recycled, and the GICS sector is today's
snapshot, not point-in-time.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = "equity-signal-research/1.0 (educational research project)"


def to_yahoo(ticker: str) -> str:
    """Wikipedia writes BRK.B, Yahoo writes BRK-B."""
    return str(ticker).strip().upper().replace(".", "-")


def fetch_wikipedia_html(cache_file: Path, max_retries: int = 4) -> str:
    """Download the page once and cache the raw html, so reruns are reproducible."""
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    import requests

    for attempt in range(max_retries):
        try:
            resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(resp.text, encoding="utf-8")
            return resp.text
        except Exception:  # noqa: BLE001 - retry any network error
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _flat_columns(df: pd.DataFrame) -> list[str]:
    if isinstance(df.columns, pd.MultiIndex):
        return [" ".join(dict.fromkeys(str(x) for x in col)).lower() for col in df.columns]
    return [str(c).lower() for c in df.columns]


def _find(columns: list[str], *words: str) -> str:
    for col in columns:
        if all(w in col for w in words):
            return col
    raise KeyError(f"no column containing {words} in {columns}")


def _tables(html: str) -> list[pd.DataFrame]:
    return pd.read_html(io.StringIO(html))


def parse_constituents(html: str) -> pd.DataFrame:
    """Current constituents: ticker, security, sector, date_added."""
    raw = next(t for t in _tables(html) if any("gics sector" in c for c in _flat_columns(t)))
    raw.columns = _flat_columns(raw)
    cols = list(raw.columns)
    added = raw[_find(cols, "date added")] if any("date added" in c for c in cols) else pd.NaT
    return pd.DataFrame({
        "ticker": raw[_find(cols, "symbol")].map(to_yahoo),
        "security": raw[_find(cols, "security")],
        "sector": raw[_find(cols, "gics sector")],
        "date_added": pd.to_datetime(added, errors="coerce"),
    })


def parse_changes(html: str, aliases: dict[str, str] | None = None) -> pd.DataFrame | None:
    """Change log: date, ticker, action ('added' or 'removed'). None if the page has no change log."""
    aliases = {to_yahoo(k): to_yahoo(v) for k, v in (aliases or {}).items()}
    raw = next((t for t in _tables(html) if any("removed" in c for c in _flat_columns(t))), None)
    if raw is None:
        return None
    raw.columns = _flat_columns(raw)
    cols = list(raw.columns)
    dates = pd.to_datetime(raw[_find(cols, "date")], errors="coerce")
    rows = []
    for action in ("added", "removed"):
        tick = raw[_find(cols, action, "ticker")]
        rows.append(pd.DataFrame({"date": dates, "ticker": tick, "action": action}).dropna())
    changes = pd.concat(rows, ignore_index=True)
    changes["ticker"] = changes["ticker"].map(to_yahoo).replace(aliases)
    changes["approximated"] = False
    return changes.sort_values(["date", "action"]).reset_index(drop=True)


def parse_wikipedia_tables(html: str, aliases: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (constituents, changes) from a page that holds both tables."""
    changes = parse_changes(html, aliases)
    if changes is None:
        raise ValueError("this page has no change log table")
    return parse_constituents(html), changes


def fetch_revision_with_changes(cache_file: Path, max_months: int = 72) -> tuple[str, pd.Timestamp]:
    """Find the most recent monthly revision of the article that still has the change log.

    The change log was removed from the live article. Wikipedia keeps every old
    revision, so we step back one month at a time until the table is there.
    Returns (html, timestamp of that revision). Cached after the first call.
    """
    import json

    import requests

    meta_file = cache_file.with_suffix(".json")
    if cache_file.exists() and meta_file.exists():
        return cache_file.read_text(encoding="utf-8"), pd.Timestamp(json.loads(meta_file.read_text())["timestamp"])
    headers = {"User-Agent": USER_AGENT}
    title = "List of S&P 500 companies"
    when = pd.Timestamp.now("UTC").normalize()
    for _ in range(max_months):
        when -= pd.DateOffset(months=1)
        reply = requests.get("https://en.wikipedia.org/w/api.php", headers=headers, timeout=30, params={
            "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "rvdir": "older",
            "rvstart": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "rvprop": "ids|timestamp", "format": "json",
            "formatversion": 2}).json()
        rev = reply["query"]["pages"][0]["revisions"][0]
        html = requests.get("https://en.wikipedia.org/w/index.php", headers=headers, timeout=60,
                            params={"title": title, "oldid": rev["revid"]}).text
        found = parse_changes(html) is not None
        print(f"  revision {rev['revid']} of {rev['timestamp'][:10]}: change log {'found' if found else 'absent'}")
        if found:
            cache_file.write_text(html, encoding="utf-8")
            meta_file.write_text(json.dumps({"revid": rev["revid"], "timestamp": rev["timestamp"]}))
            return html, pd.Timestamp(rev["timestamp"])
        time.sleep(1.0)
    raise RuntimeError("no revision with a change log found")


def bridge_gap(current: pd.DataFrame, snapshot: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Approximate change events between an old snapshot of the index and today's list.

    Additions (in today's list, not in the snapshot) are dated by the 'Date
    added' column. Removals (in the snapshot, not in today's list) have no
    known date. They are dated at the last addition in the gap: the latest
    plausible date, so no name is dropped early using knowledge of its later
    removal. All these rows are marked approximated=True and reported.
    """
    snapshot_date = pd.Timestamp(snapshot_date).tz_localize(None).normalize()
    new = current[~current["ticker"].isin(snapshot["ticker"])]
    added_dates = new["date_added"].where(new["date_added"] > snapshot_date, snapshot_date + pd.Timedelta(days=1))
    added = pd.DataFrame({"date": added_dates.to_numpy(), "ticker": new["ticker"].to_numpy(), "action": "added"})
    gone = snapshot.loc[~snapshot["ticker"].isin(current["ticker"]), "ticker"]
    last = added["date"].max() if len(added) else snapshot_date + pd.Timedelta(days=1)
    removed = pd.DataFrame({"date": last, "ticker": gone.to_numpy(), "action": "removed"})
    out = pd.concat([added, removed], ignore_index=True)
    out["approximated"] = True
    return out


GITHUB_HISTORY_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
                      "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def fetch_github_history(cache_file: Path) -> pd.DataFrame:
    """Public dataset of dated S&P 500 member lists (github.com/fja05680/sp500), cached.

    It is compiled by its author from a published list (1996-2019) plus the
    Wikipedia change log since then. Columns: date, tickers (comma separated).
    """
    if not cache_file.exists():
        import requests

        resp = requests.get(GITHUB_HISTORY_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        cache_file.write_text(resp.text, encoding="utf-8")
    return pd.read_csv(cache_file, parse_dates=["date"])


def snapshots_to_changes(history: pd.DataFrame, aliases: dict[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn dated member lists into add/remove events by differencing consecutive lists.

    Returns (changes, last snapshot as a one-column 'ticker' frame).
    """
    aliases = {to_yahoo(k): to_yahoo(v) for k, v in (aliases or {}).items()}
    history = history.sort_values("date")
    sets = [{aliases.get(to_yahoo(t), to_yahoo(t)) for t in row.split(",") if t.strip()} for row in history["tickers"]]
    rows = []
    for date, before, after in zip(history["date"].iloc[1:], sets[:-1], sets[1:]):
        rows += [{"date": date, "ticker": t, "action": "added"} for t in sorted(after - before)]
        rows += [{"date": date, "ticker": t, "action": "removed"} for t in sorted(before - after)]
    changes = pd.DataFrame(rows, columns=["date", "ticker", "action"])
    changes["approximated"] = False
    return changes, pd.DataFrame({"ticker": sorted(sets[-1])})


def load_universe_from_wikipedia(universe_dir: Path, aliases: dict[str, str] | None, max_retries: int = 4,
                                 source: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (current constituents, full change log, source info).

    Current constituents and sectors always come from the live Wikipedia table.
    The change log comes from, in order (source='auto'):
      1. the live article, if it still has the table;
      2. the latest old revision of the article that has it;
      3. the public GitHub dataset of historical member lists.
    For 2 and 3, changes after the snapshot date are bridged (see bridge_gap).
    """
    current_html = fetch_wikipedia_html(universe_dir / "sp500_wikipedia.html", max_retries)
    constituents = parse_constituents(current_html)
    changes = parse_changes(current_html, aliases) if source in ("auto", "wikipedia") else None
    if changes is not None:
        return constituents, changes, {"change_log_source": "live Wikipedia article", "snapshot_date": None,
                                       "n_approximated_events": 0}
    label, base, snapshot, stamp = None, None, None, None
    if source in ("auto", "wikipedia"):
        try:
            old_html, stamp = fetch_revision_with_changes(universe_dir / "sp500_wikipedia_changes.html")
            base, snapshot = parse_changes(old_html, aliases), parse_constituents(old_html)
            label = "old revision of the Wikipedia article"
        except Exception as exc:  # noqa: BLE001
            if source == "wikipedia":
                raise
            print(f"  Wikipedia revision search failed ({exc!r}); using the GitHub dataset")
    if base is None:
        history = fetch_github_history(universe_dir / "sp500_github_history.csv")
        base, snapshot = snapshots_to_changes(history, aliases)
        stamp, label = history["date"].max(), "GitHub dataset fja05680/sp500 (historical member lists)"
    gap = bridge_gap(constituents, snapshot, stamp)
    changes = pd.concat([base, gap], ignore_index=True).sort_values(["date", "action"]).reset_index(drop=True)
    return constituents, changes, {"change_log_source": label, "snapshot_date": str(pd.Timestamp(stamp).date()),
                                   "n_approximated_events": int(len(gap))}


def build_intervals(
    current: list[str], changes: pd.DataFrame, data_start: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict]]:
    """Membership intervals [start, end) per ticker, plus a list of inconsistencies.

    An inconsistency is an event that does not fit the walked-back set, e.g.
    an 'added' ticker that is not a member just after its add date (usually a
    later rename). They are logged so the reader can judge the data quality.
    """
    far_future = pd.Timestamp("2262-01-01")
    open_end: dict[str, pd.Timestamp] = {t: far_future for t in current}
    intervals: list[dict] = []
    issues: list[dict] = []
    for date, day in sorted(changes.groupby("date"), key=lambda kv: kv[0], reverse=True):
        if date < data_start:
            break
        for ticker in day.loc[day["action"] == "added", "ticker"]:
            if ticker in open_end:
                intervals.append({"ticker": ticker, "start": date, "end": open_end.pop(ticker)})
            else:
                issues.append({"date": date, "ticker": ticker, "issue": "added but not a member afterwards"})
        for ticker in day.loc[day["action"] == "removed", "ticker"]:
            if ticker in open_end:
                issues.append({"date": date, "ticker": ticker, "issue": "removed but still a member afterwards"})
            else:
                open_end[ticker] = date
    for ticker, end in open_end.items():
        intervals.append({"ticker": ticker, "start": data_start, "end": end})
    out = pd.DataFrame(intervals).sort_values(["ticker", "start"]).reset_index(drop=True)
    return out, issues


def membership_matrix(intervals: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean dates x tickers matrix: True when start <= date < end."""
    member = pd.DataFrame(False, index=dates, columns=tickers)
    for row in intervals.itertuples(index=False):
        if row.ticker in member.columns:
            member.loc[(dates >= row.start) & (dates < row.end), row.ticker] = True
    return member
