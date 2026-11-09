"""Topic finder: what "What If" premises are pulling views right now, and which are already taken.

Usage:  venv\\Scripts\\python topics.py [extra search terms ...]
Reads YouTube search result pages (no API key), keeps videos from the last ~60 days, and writes
research/topics_<date>.md: premises ranked by views per day, plus every title seen (so a new episode avoids taken angles).
"""
import datetime
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
SEEDS = ["what if you could", "what if you were", "what if you had", "what if you became", "what if you could see",
         "what if you woke up", "what if animation story", "what if stickman animation", "your life if you could",
         "what if you never", "what if every human", "what if you lived"]
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
      "Accept-Language": "en-US,en;q=0.9"}
UNITS = {"minute": 1 / 1440, "hour": 1 / 24, "day": 1, "week": 7, "month": 30, "year": 365}


def search(query, recent=True):
    # sp=EgIIBA%3D%3D: uploaded this month; sp=CAMSAhAB: sort by views (all time)
    sp = "EgQIBBAB" if recent else "CAMSAhAB"
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}&sp={sp}"
    for _ in range(3):  # YouTube sometimes serves a page without the data block; a retry gets the normal one
        html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read().decode("utf-8", "replace")
        m = re.search(r"var ytInitialData = (\{.*?\});</script>", html, re.S)
        if m:
            break
    else:
        print(f"skipped {query!r}: no results page")
        return []
    vids = []

    def walk(o):
        if isinstance(o, dict):
            r = o.get("videoRenderer")
            if r:
                vids.append(r)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(m.group(1)))
    out = []
    for r in vids:
        views = re.sub(r"\D", "", (r.get("viewCountText") or {}).get("simpleText", "") or "0") or "0"
        age = (r.get("publishedTimeText") or {}).get("simpleText", "")
        am = re.match(r"(?:Streamed )?(\d+)\s*(months?|mo|minutes?|mins?|m|hours?|h|days?|d|weeks?|w|years?|y)\b", age)
        if not am:
            continue
        unit = {"mo": "month", "m": "minute", "min": "minute", "h": "hour", "d": "day", "w": "week", "y": "year"}.get(
            am.group(2).rstrip("s") if len(am.group(2)) > 2 else am.group(2), am.group(2).rstrip("s"))
        days = max(0.5, int(am.group(1)) * UNITS[unit])
        out.append({"id": r["videoId"], "title": r["title"]["runs"][0]["text"], "channel": r["ownerText"]["runs"][0]["text"],
                    "views": int(views), "days": days, "len": (r.get("lengthText") or {}).get("simpleText", "")})
    return out


def premise(title):
    """'What IF You Could See Everyone's Future Wealth? Part 3' -> 'could see everyone's future wealth'"""
    t = title.lower()
    t = re.sub(r"\(.*?\)|\[.*?\]|part \d+|full movie|#\S+", "", t)
    t = re.sub(r"^.*?what if (you |your )?", "", t)
    return re.sub(r"[^a-z0-9' ]", "", t).strip()


def main(extra):
    seen = {}
    for q in SEEDS + extra:
        for v in search(q):
            if v["days"] <= 60 and v["id"] not in seen:
                seen[v["id"]] = v
    vids = sorted(seen.values(), key=lambda v: v["views"] / v["days"], reverse=True)
    groups = {}
    for v in vids:
        groups.setdefault(premise(v["title"]), []).append(v)
    today = datetime.date.today().isoformat()
    lines = [f"# What If topics, {today}", "", f"{len(vids)} videos from the last 60 days.", "",
             "## Premises by views per day (best video)", "", "| views/day | views | age (d) | channel | title |", "|---|---|---|---|---|"]
    for p, vs in sorted(groups.items(), key=lambda kv: -max(v["views"] / v["days"] for v in kv[1]))[:60]:
        v = max(vs, key=lambda v: v["views"] / v["days"])
        lines.append(f"| {int(v['views'] / v['days']):,} | {v['views']:,} | {v['days']:.0f} | {v['channel']} | {v['title']} |"
                     + (f" ({len(vs)} channels)" if len(vs) > 1 else ""))
    lines += ["", "## Every premise seen (taken angles)", ""] + [f"- {p}" for p in sorted(groups)]
    out = ROOT / "research" / f"topics_{today}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(vids)} videos, {len(groups)} premises -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
