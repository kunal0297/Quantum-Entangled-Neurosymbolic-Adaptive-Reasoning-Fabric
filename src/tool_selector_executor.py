from __future__ import annotations

from typing import Any, Dict, Tuple

import networkx as nx
import sympy as sp


def _extract_numbers(text: str):
    """Extract all numbers from text."""
    import re
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

def _safe_eval_arithmetic(expr: str) -> Tuple[Any, str]:
    try:
        # First try direct sympy evaluation
        res = sp.simplify(sp.sympify(expr))
        return res, "ok"
    except Exception as e:
        return None, f"error: {e}"

def _detect_operation(text: str):
    """Detect mathematical operation from text."""
    text_lower = text.lower()
    
    # Addition
    if any(op in text_lower for op in ["plus", "add", "+", "sum"]):
        return "add"
    
    # Subtraction  
    if any(op in text_lower for op in ["minus", "subtract", "-", "difference"]):
        return "subtract"
    
    # Multiplication
    if any(op in text_lower for op in ["times", "multiply", "*", "product"]):
        return "multiply"
    
    # Division
    if any(op in text_lower for op in ["divide", "divided by", "/", "over"]):
        return "divide"
    
    # Comparison
    if any(op in text_lower for op in ["greater", "larger", "bigger", ">"]):
        return "greater"
    
    if any(op in text_lower for op in ["less", "smaller", "<"]):
        return "less"
    
    # Even/odd
    if "even" in text_lower:
        return "even"
    if "odd" in text_lower:
        return "odd"
    
    return None


def execute_node(node: str, g: nx.DiGraph, context: Dict[str, Any]) -> Dict[str, Any]:
    kind = g.nodes[node].get("kind")
    text = g.nodes["input"].get("text", "")

    if kind == "tool":
        # Extract numbers and operation from text
        numbers = _extract_numbers(text)
        operation = _detect_operation(text)
        
        if node == "math_solve" or node == "arithmetic":
            # Handle equations first
            if "=" in text:
                try:
                    left, right = text.split("=", 1)
                    left_clean = left.strip().replace("x", "*x").replace("X", "*X")
                    right_clean = right.strip()
                    res = sp.solve(sp.Eq(sp.sympify(left_clean), sp.sympify(right_clean)))
                    if res:
                        return {"result": res[0] if len(res) == 1 else res, "node": node}
                except Exception:
                    pass
            
            # Handle arithmetic operations
            if len(numbers) >= 2 and operation:
                try:
                    if operation == "add":
                        result = sum(numbers)
                    elif operation == "subtract":
                        result = numbers[0] - numbers[1]
                    elif operation == "multiply":
                        result = numbers[0] * numbers[1] 
                    elif operation == "divide":
                        result = numbers[0] / numbers[1] if numbers[1] != 0 else None
                    else:
                        result = None
                    
                    if result is not None:
                        return {"result": int(result) if result.is_integer() else result, "node": node}
                except Exception:
                    pass
            
            # Try direct sympy evaluation as fallback
            try:
                # Clean up text for sympy
                clean_text = text.replace("what is", "").replace("calculate", "").replace("compute", "").strip()
                clean_text = clean_text.replace("plus", "+").replace("times", "*").replace("minus", "-")
                clean_text = clean_text.replace("divided by", "/").replace("?", "").strip()
                
                if clean_text:
                    result = sp.sympify(clean_text)
                    return {"result": float(result) if result.is_number else result, "node": node}
            except Exception:
                pass

        if node == "compare":
            if len(numbers) >= 2 and operation:
                a, b = numbers[0], numbers[1]
                if operation == "greater":
                    return {"result": a if a > b else b, "node": node}
                elif operation == "less":
                    return {"result": a if a < b else b, "node": node}
            return {"result": None, "node": node}

        if node == "logic_branch":
            # Handle even/odd checks
            if operation == "even" and numbers:
                num = int(numbers[0])
                return {"result": num % 2 == 0, "node": node}
            elif operation == "odd" and numbers:
                num = int(numbers[0])
                return {"result": num % 2 == 1, "node": node}
            
            # Handle logical statements
            if " if " in text.lower() and " then " in text.lower():
                return {"result": True, "node": node, "explanation": "Implication pattern detected"}
            if any(k in text.lower() for k in ["true", "always", "must"]):
                return {"result": True, "node": node}
            if any(k in text.lower() for k in ["false", "never", "cannot"]):
                return {"result": False, "node": node}
            return {"result": None, "node": node}

        if node == "fallback":
            # Fallback should try to handle any simple math
            if numbers and operation:
                if operation == "add" and len(numbers) >= 2:
                    return {"result": sum(numbers), "node": node}
                elif operation == "multiply" and len(numbers) >= 2:
                    return {"result": numbers[0] * numbers[1], "node": node}
            return {"result": None, "node": node}

    if kind == "aggregate":
        preds = list(g.predecessors(node))
        for p in preds:
            res = context.get(p, {}).get("result")
            if res is not None:
                return {"result": res, "node": node}
        return {"result": None, "node": node}

    if kind == "verify":
        preds = list(g.predecessors(node))
        res = context.get(preds[0], {}).get("result") if preds else None
        return {"result": res, "node": node}

    if kind == "input":
        return {"result": text, "node": node}

    return {"result": None, "node": node}


def execute_graph_topologically(g: nx.DiGraph) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    try:
        # Check if graph is acyclic
        topo_order = list(nx.topological_sort(g))
        for node in topo_order:
            ctx[node] = execute_node(node, g, ctx)
    except nx.NetworkXError:
        # Handle cycles by executing in arbitrary order
        print("Warning: Graph contains cycles, executing in arbitrary order")
        for node in g.nodes():
            try:
                ctx[node] = execute_node(node, g, ctx)
            except Exception as e:
                print(f"Error executing node {node}: {e}")
                ctx[node] = {"result": None, "node": node, "error": str(e)}
    return ctx


