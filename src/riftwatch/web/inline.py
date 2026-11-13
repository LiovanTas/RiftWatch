"""The inline scripts pages carry, in one place, so the content security policy can allow
exactly these (by hash) and no other inline script: markup injected into a page couldn't run
a script of its own even if something were ever left unescaped.

Each script is fixed text. Pages render them from these strings and the policy hashes the
same strings, so the two can't drift apart.
"""

from __future__ import annotations

import base64
import hashlib

from riftwatch.report.html import _JS, ACTION_JS

# Before the stylesheet paints: the visitor's saved theme, else their system's.
THEME_JS = ("try{var t=JSON.parse(localStorage.getItem('rw-theme'));if(t)document.documentElement.dataset.theme=t;"
            "else if(matchMedia('(prefers-color-scheme: light)').matches)document.documentElement.dataset.theme='light'}"
            "catch(e){}")
# The analysis pages (report.html) wrapped in the site's header: saved theme, else dark.
REPORT_THEME_JS = ('try{var t=JSON.parse(localStorage.getItem("rw-theme"));'
                   'document.documentElement.dataset.theme=t||"dark"}catch(e){'
                   'document.documentElement.dataset.theme="dark"}')
REPORT_JS = _JS + ACTION_JS              # report.html's page script (charts, coach buttons)
INLINE_SCRIPTS = (THEME_JS, REPORT_THEME_JS, ACTION_JS, REPORT_JS)


def _hash(text: str) -> str:
    return "'sha256-" + base64.b64encode(hashlib.sha256(text.encode("utf-8")).digest()).decode() + "'"


SCRIPT_HASHES = " ".join(_hash(t) for t in INLINE_SCRIPTS)
