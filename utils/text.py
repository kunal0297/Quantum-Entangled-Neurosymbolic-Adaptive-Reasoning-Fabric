from __future__ import annotations

import re


def normalize_answer(ans) -> str:
    s = str(ans)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


