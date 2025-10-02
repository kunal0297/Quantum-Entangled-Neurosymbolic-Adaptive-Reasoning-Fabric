from __future__ import annotations

from typing import Any, Dict, Tuple

import sympy as sp
import networkx as nx


def symbolic_numeric_verification(value: Any) -> Tuple[float, str]:
    try:
        if value is None:
            return 0.0, "none"
        if isinstance(value, (int, float)):
            return 1.0, "numeric"
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return 0.9, "list"
        s = sp.sympify(value)
        if s.is_number:
            return 1.0, "sympy_number"
        return 0.5, "sympy_expr"
    except Exception as e:
        return 0.0, f"error:{e}"


def verification_score(g: nx.DiGraph, ctx: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    result = None
    if g.has_node("verify"):
        preds = list(g.predecessors("verify"))
        if preds:
            result = ctx.get(preds[0], {}).get("result")
    else:
        result = ctx.get("aggregate", {}).get("result")

    score, mode = symbolic_numeric_verification(result)
    efficiency = max(0.0, 1.0 - 0.02 * (g.number_of_nodes() + g.number_of_edges()))
    return score * 0.8 + efficiency * 0.2, {"verification_mode": mode, "efficiency": efficiency, "result": result}


