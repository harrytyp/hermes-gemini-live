"""What this call cost, in numbers only — and honest about not knowing.

Adopted from hermes-live's usage ledger (dandacompany), cut down to one provider: a record
per call with duration and server-reported tokens, an estimated dollar figure from a dated
price table, and an ``unpriced`` flag whenever the report is incomplete (cached content,
thought tokens, tool-use tokens, or a model missing from the table).

Two rules that matter more than the arithmetic:

* **only numbers are stored** — no audio, no transcript, no key, no device id;
* **an unknown price is said out loud** rather than guessed at, so the panel can print
  "unpriced" next to a figure instead of quietly under-reporting a bill.

The figure is an estimate from published rates, not a bill: free tiers, discounts, taxes
and the cost of delegated Hermes work are not in it.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

PRICING_DATE = "2026-11-04"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"

#: USD per 1M tokens: (input text, input audio, output text, output audio). Keyed by the
#: model id prefix the endpoint reports; a model that is not in here is priced as unknown.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "gemini-3.8-live": (0.75, 3.0, 4.5, 12.0),
    "gemini-3.1-flash-live": (0.75, 3.0, 4.5, 12.0),
    "gemini-2.5-flash-native-audio": (0.5, 3.0, 2.0, 12.0),
}
MAX_RECORDS = 2000
KEEP_DAYS = 31


def _home() -> Path:
    from . import settings
    return settings._home()


def ledger_path() -> Path:
    return _home() / "plugin-data" / "hermes-gemini-live" / "usage.json"


def price_for(model: str) -> tuple[float, float, float, float] | None:
    for prefix, price in PRICES.items():
        if model.startswith(prefix):
            return price
    return None


def _count(value) -> int:
    return int(value) if isinstance(value, (int, float)) and value >= 0 else 0


def _modalities(rows) -> dict:
    """``promptTokensDetails`` / ``responseTokensDetails`` summed per modality."""
    sums = {"AUDIO": 0, "TEXT": 0}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("modality") in sums:
                sums[row["modality"]] += _count(row.get("tokenCount"))
    return sums


class Usage:
    """One call's numbers, folded into a rolling local ledger."""

    def __init__(self, model: str, started: float | None = None) -> None:
        self.id = uuid.uuid4().hex
        self.model = model
        self.start = started or time.time()
        self.end = self.start
        self.input = 0
        self.output = 0
        self.usd = 0.0
        self.unpriced = price_for(model) is None
        self.reports = 0

    def tokens(self, usage: dict | None) -> None:
        """Fold one ``usageMetadata`` frame in. Server numbers only, never derived guesses.

        The endpoint reports cumulative totals for the session, so every report *replaces*
        the running figures rather than adding to them.
        """
        if not isinstance(usage, dict):
            return
        prompt = _count(usage.get("promptTokenCount"))
        response = _count(usage.get("candidatesTokenCount") or usage.get("responseTokenCount"))
        if not prompt and not response:
            return
        self.input = max(self.input, prompt)
        self.output = max(self.output, response)
        self.reports += 1
        price = price_for(self.model)
        if price is None:
            self.unpriced = True
            return
        # Live traffic is audio, so a report without modality details cannot be priced:
        # charging it at text rates would quietly under-report every call.
        hinted = any(_count(usage.get(key)) for key in
                     ("cachedContentTokenCount", "thoughtsTokenCount",
                      "toolUsePromptTokenCount"))
        in_detail = _modalities(usage.get("promptTokensDetails"))
        out_detail = _modalities(usage.get("responseTokensDetails"))
        complete = (not hinted and self.input + self.output > 0
                    and sum(in_detail.values()) == self.input
                    and sum(out_detail.values()) == self.output)
        if not complete:
            self.unpriced = True
            return
        in_text, in_audio, out_text, out_audio = price
        self.usd = round((in_detail["TEXT"] * in_text + in_detail["AUDIO"] * in_audio
                          + out_detail["TEXT"] * out_text
                          + out_detail["AUDIO"] * out_audio) / 1e6, 6)
        self.unpriced = False

    def close(self) -> dict:
        self.end = max(self.end, time.time())
        return to_record(self)


def to_record(call: Usage | dict) -> dict:
    data = call.__dict__ if isinstance(call, Usage) else dict(call)
    start = float(data.get("start") or 0)
    end = float(data.get("end") or start)
    return {"id": data.get("id") or uuid.uuid4().hex, "model": data.get("model") or "",
            "start": start, "end": end, "duration_ms": int(max(0.0, end - start) * 1000),
            "input": int(data.get("input") or 0), "output": int(data.get("output") or 0),
            "usd": float(data.get("usd") or 0.0), "unpriced": bool(data.get("unpriced")),
            "reports": int(data.get("reports") or 0), "price_date": PRICING_DATE}


def _read() -> dict:
    try:
        return json.loads(ledger_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"records": [], "archived": {}}
    except (OSError, ValueError) as exc:
        logger.info("hermes-gemini-live: usage ledger unreadable (%s); starting empty",
                    type(exc).__name__)
        return {"records": [], "archived": {}}


def _write(data: dict) -> None:
    path = ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp = tempfile.mkstemp(prefix="usage-", suffix=".json", dir=str(path.parent))
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, path)
    except OSError as exc:
        logger.info("hermes-gemini-live: usage not recorded (%s)", type(exc).__name__)


def record(call: Usage | dict) -> None:
    """Append one finished call, folding old records into archived totals."""
    entry = to_record(call)
    data = _read()
    records = [row for row in data.get("records") or [] if row.get("id") != entry["id"]]
    records.append(entry)
    archived = dict(data.get("archived") or {})
    cutoff = time.time() - KEEP_DAYS * 86400
    while records and (len(records) > MAX_RECORDS or float(records[0].get("end") or 0) < cutoff):
        old = records.pop(0)
        key = str(old.get("model") or "unknown")
        bucket = dict(archived.get(key) or {"calls": 0, "duration_ms": 0, "input": 0,
                                            "output": 0, "usd": 0.0, "unpriced": 0})
        bucket["calls"] += 1
        bucket["duration_ms"] += int(old.get("duration_ms") or 0)
        bucket["input"] += int(old.get("input") or 0)
        bucket["output"] += int(old.get("output") or 0)
        bucket["usd"] = round(float(bucket["usd"]) + float(old.get("usd") or 0), 6)
        bucket["unpriced"] += int(bool(old.get("unpriced")))
        archived[key] = bucket
    _write({"records": records, "archived": archived})


def summary(since_seconds: int = 86400) -> dict:
    """Totals for the window and for all time, plus the newest calls for the panel."""
    data = _read()
    now = time.time()
    since = now - since_seconds
    window = {"calls": 0, "duration_ms": 0, "input": 0, "output": 0, "usd": 0.0, "unpriced": 0}
    lifetime = {"calls": 0, "duration_ms": 0, "input": 0, "output": 0, "usd": 0.0, "unpriced": 0}
    for row in data.get("records") or []:
        target = [lifetime]
        if float(row.get("end") or 0) >= since:
            target.append(window)
        for bucket in target:
            bucket["calls"] += 1
            bucket["duration_ms"] += int(row.get("duration_ms") or 0)
            bucket["input"] += int(row.get("input") or 0)
            bucket["output"] += int(row.get("output") or 0)
            bucket["usd"] = round(bucket["usd"] + float(row.get("usd") or 0), 6)
            bucket["unpriced"] += int(bool(row.get("unpriced")))
    for bucket in (data.get("archived") or {}).values():
        lifetime["calls"] += int(bucket.get("calls") or 0)
        lifetime["duration_ms"] += int(bucket.get("duration_ms") or 0)
        lifetime["input"] += int(bucket.get("input") or 0)
        lifetime["output"] += int(bucket.get("output") or 0)
        lifetime["usd"] = round(lifetime["usd"] + float(bucket.get("usd") or 0), 6)
        lifetime["unpriced"] += int(bucket.get("unpriced") or 0)
    recent = sorted((data.get("records") or []), key=lambda r: float(r.get("end") or 0),
                    reverse=True)[:10]
    return {"last_24h": window, "all_time": lifetime, "recent": recent,
            "price_date": PRICING_DATE, "price_url": PRICING_URL}
