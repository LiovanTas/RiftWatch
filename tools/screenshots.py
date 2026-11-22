"""Headless screenshots of RiftWatch pages, so template and CSS changes can be looked at
without opening a browser window.

    .venv\\Scripts\\python tools\\screenshots.py / /champions /players/na/Liovan-G2EU
    .venv\\Scripts\\python tools\\screenshots.py --url http://127.0.0.1:8000 /swing --theme both

Without --url it starts the site from this checkout on a free port, so the shots show the code
as it is now (the database must be up). Each page is shot full length at desktop and phone width
into out/shots/<page>-<size>-<theme>.png, and the run lists console errors, failed requests and
anything that makes the page scroll sideways. Exit code 1 if any page had a problem.

One-time setup: pip install -e ".[shots]" and python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"desktop": (1440, 900), "phone": (390, 844)}

# When the page scrolls sideways: the elements that stick out past the right edge while their
# parent doesn't (the cause, not everything inside it), skipping ones inside a scroll box.
OVERFLOW_JS = """() => {
  const w = document.documentElement.clientWidth;
  const scroll = document.documentElement.scrollWidth - w;
  if (scroll <= 1) return {scroll: 0, worst: []};
  const sticksOut = el => el.getBoundingClientRect().right > w + 1;
  const clipped = el => {
    for (let p = el.parentElement; p && p !== document.body; p = p.parentElement)
      if (getComputedStyle(p).overflowX !== 'visible') return true;
    return false;
  };
  const worst = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (!r.width || !sticksOut(el)) continue;
    if (el.parentElement !== document.body && sticksOut(el.parentElement)) continue;
    if (clipped(el)) continue;
    const cls = typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\\s+/).join('.') : '';
    worst.push([Math.round(r.right - w), el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + cls]);
  }
  worst.sort((a, b) => b[0] - a[0]);
  return {scroll, worst: worst.slice(0, 5)};
}"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_site():
    """The site from this checkout, in this process, on a free port."""
    import uvicorn
    from dotenv import load_dotenv

    from riftwatch.web.app import create_app

    load_dotenv(ROOT / ".env")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port,
                                           log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, name="site", daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise SystemExit("error: the site didn't start (is the database up?)")
        time.sleep(0.2)
    return f"http://127.0.0.1:{port}", server, thread


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", path).strip("-")[:80] or "home"


def _shoot(page, url: str, file: Path, wait_s: float) -> list[str]:
    problems: list[str] = []
    page.on("console", lambda m: m.type == "error" and problems.append(f"console: {m.text}"))
    page.on("pageerror", lambda e: problems.append(f"script error: {e}"))
    page.on("requestfailed", lambda r: problems.append(f"request failed: {r.url} ({r.failure})"))
    page.on("response", lambda r: r.status >= 400 and problems.append(f"HTTP {r.status}: {r.url}"))
    page.goto(url, wait_until="load", timeout=90_000)
    try:                                   # pages that poll never go idle; that's fine
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    page.wait_for_timeout(wait_s * 1000)
    over = page.evaluate(OVERFLOW_JS)
    if over["scroll"]:
        causes = ", ".join(f"{sel} (+{px})" for px, sel in over["worst"]) or "unknown element"
        problems.append(f"scrolls sideways by {over['scroll']} px: {causes}")
    page.screenshot(path=str(file), full_page=True, animations="disabled")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headless screenshots of RiftWatch pages.")
    ap.add_argument("paths", nargs="*", default=["/"],
                    help="site paths, e.g. / /champions /players/na/Name-TAG (default /)")
    ap.add_argument("--url", help="a running site, e.g. http://127.0.0.1:8000 "
                                  "(default: start one from this checkout)")
    ap.add_argument("--size", choices=[*SIZES, "both"], default="both")
    ap.add_argument("--theme", choices=["dark", "light", "both"], default="dark")
    ap.add_argument("--wait", type=float, default=0.5,
                    help="seconds to wait after the page settles (default 0.5)")
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "shots")
    args = ap.parse_args(argv)
    windows_path = next((p for p in args.paths if re.match(r"^[A-Za-z]:[\\/]", p)), None)
    if windows_path:                       # Git Bash turns "/champions" into "C:/Program Files/Git/champions"
        print(f"error: {windows_path!r} is a Windows path, not a site path; from Git Bash run "
              "this with MSYS_NO_PATHCONV=1 in front", file=sys.stderr)
        return 2

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print('error: needs Playwright: pip install -e ".[shots]", then '
              "python -m playwright install chromium", file=sys.stderr)
        return 2

    sizes = SIZES if args.size == "both" else {args.size: SIZES[args.size]}
    themes = ["dark", "light"] if args.theme == "both" else [args.theme]
    args.out.mkdir(parents=True, exist_ok=True)
    server = thread = None
    if args.url:
        base = args.url.rstrip("/")
    else:
        print("starting the site from this checkout (~20 s to load the models)...", flush=True)
        base, server, thread = _start_site()

    failed = 0
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()          # headless: no window opens
            for theme in themes:
                for size, (width, height) in sizes.items():
                    phone = size == "phone"
                    context = browser.new_context(viewport={"width": width, "height": height},
                                                  color_scheme=theme, is_mobile=phone,
                                                  has_touch=phone)
                    for path in args.paths:
                        file = args.out / f"{_slug(path)}-{size}-{theme}.png"
                        page = context.new_page()
                        try:
                            problems = _shoot(page, base + "/" + path.lstrip("/"), file, args.wait)
                        except Exception as e:     # one bad page shouldn't stop the rest
                            problems = [f"couldn't load: {e}".splitlines()[0]]
                        finally:
                            page.close()
                        failed += bool(problems)
                        print(f"{file.relative_to(ROOT) if file.is_relative_to(ROOT) else file}  "
                              f"{'ok' if not problems else f'{len(problems)} problem(s)'}")
                        for line in dict.fromkeys(problems):
                            print(f"    {line}")
                    context.close()
            browser.close()
    finally:
        if server is not None:
            server.should_exit = True
            thread.join(timeout=15)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
