from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import sympy as sp
import networkx as nx
import math 


def symbolic_numeric_verification(value: Any) -> Tuple[float, str]:
    try:
        if value is None:
            return 0.0, "none"
        
        if isinstance(value, bool):
            return 0.98, "boolean"
        
        if isinstance(value, (int, float)):
            return 1.0, "numeric_final"
            
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return 0.1, "empty_list"
            first_val = value[0]
            if isinstance(first_val, (int, float, bool)):
                return 0.95, "list_of_results"
            
        s = sp.sympify(value, evaluate=False) 
        if s.is_number:
            return 1.0, "sympy_number"
        
        return 0.6, "sympy_expr"
        
    except Exception as e:
        return 0.05, f"error:{type(e).__name__}"


def verification_score(g: nx.DiGraph, ctx: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    
    result = None
    if g.has_node("verify"):
        preds = list(g.predecessors("verify"))
        if preds:
            result = ctx.get(preds[0], {}).get("result")
    else:
        result = ctx.get("aggregate", {}).get("result")
        
    base_score, mode = symbolic_numeric_verification(result)
    
    raw_confidence = g.nodes.get("verify", {}).get("raw_confidence", 0.5)

    complexity = g.number_of_nodes() + g.number_of_edges()
    
    efficiency = max(0.0, 1.0 - 0.15 * math.log(1 + complexity))
    
    calibrated_confidence = (base_score * 0.7) + (raw_confidence * 0.3)
    
    final_score = (calibrated_confidence * 0.85) + (efficiency * 0.15)
    
    final_score = max(0.0, min(1.0, final_score))

    metadata = {
        "verification_mode": mode,
        "base_score": base_score,
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "efficiency": efficiency,
        "complexity": complexity,
        "result": result
    }
    
    return final_score, metadata