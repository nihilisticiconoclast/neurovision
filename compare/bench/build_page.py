"""Put the benchmark results into compare/index.html.

Embeds summary.json (the fields the charts use) between the <script id="results"> tags, and
fills every element marked data-num="..." with its number, so the page's text always matches
the charts:

  data-num="claude-cnn mnist pct"                   99.3%
  data-num="claude-cnn mnist count"                 9,933   (digits correct)
  data-num="claude-cnn params int"                  11,274
  data-num="gemini mnist400_pred_counts 1 int"      111     (an index into a list)
  data-num="measured date"                          27 November 2026
  data-num="browser major"                          Chrome 153

The prose around the numbers is not rewritten: after a re-run, read it again.

Usage: python3 build_page.py summary.json
"""
import datetime
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "..", "index.html")
KEEP = ("params", "mnist", "mnist400_normal", "mnist400_half", "mnist400_quarter",
        "styles600_normal", "styles600_half", "styles600_quarter", "mnist400_pred_counts")


def pct(v):
    """One decimal place, exact halves to even (177 of 400 = 44.25% -> 44.2%), as the page's script does."""
    if v >= 0.9995:
        return "100%"
    t = v * 1000
    f = math.floor(t)
    r = (f if f % 2 == 0 else f + 1) if abs(t - f - 0.5) < 1e-9 else round(t)
    return f"{r / 10:.1f}%"


def number(data, spec):
    t = spec.split()
    if t == ["measured", "date"]:
        d = datetime.date.fromisoformat(data["measured"])
        return f"{d.day} {d:%B %Y}"
    if t == ["browser", "major"]:
        name, _, version = data["browser"].partition("/")
        return f"{name.replace('HeadlessChrome', 'Chrome')} {version.split('.')[0]}"
    app, key, *index, fmt = t
    v = data["apps"][app][key]
    for i in index:
        v = v[int(i)]
    if fmt == "pct":
        return pct(v)
    if fmt == "count":
        n = data["n"]["mnist400" if key.startswith("mnist400") else "styles600" if key.startswith("styles600") else "mnist"]
        return f"{round(v * n):,}"
    if fmt == "int":
        return f"{v:,}"
    raise ValueError(f"unknown format in data-num={spec!r}")


def main(summary_path):
    s = json.load(open(summary_path))
    data = {"measured": s["measured"], "browser": s["browser"], "n": s["n"],
            "apps": {app: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items() if k in KEEP}
                     for app, a in s["apps"].items()}}
    page = open(PAGE, encoding="utf-8").read()

    block = re.compile(r'(<script id="results" type="application/json">)(.*?)(</script>)', re.S)
    assert len(block.findall(page)) == 1, "expected one results block"
    page = block.sub(lambda m: m.group(1) + "\n" + json.dumps(data, separators=(",", ":")) + "\n" + m.group(3), page)

    changed = []
    marked = re.compile(r'(<(?P<tag>[a-z0-9]+)\b[^>]*\bdata-num="(?P<spec>[^"]+)"[^>]*>)(?P<text>[^<]*)(</(?P=tag)>)')

    def fill(m):
        new = number(data, m.group("spec"))
        if new != m.group("text"):
            changed.append(f'  {m.group("spec")}: {m.group("text")} -> {new}')
        return m.group(1) + new + m.group(5)

    page, filled = marked.subn(fill, page)
    open(PAGE, "w", encoding="utf-8").write(page)
    print(f"embedded results measured {data['measured']}; filled {filled} numbers in the text, {len(changed)} changed")
    for line in changed:
        print(line)


if __name__ == "__main__":
    main(sys.argv[1])
