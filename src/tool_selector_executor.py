from __future__ import annotations

from typing import Any, Dict, Tuple, List
import networkx as nx
import sympy as sp
import re
import math


def _extract_numbers(text: str):
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

def _detect_operation(text: str):
    text_lower = text.lower()
    
    if any(op in text_lower for op in ["plus", "add", "+", "sum"]): return "add"
    if any(op in text_lower for op in ["minus", "subtract", "-", "difference"]): return "subtract"
    if any(op in text_lower for op in ["times", "multiply", "*", "product"]): return "multiply"
    if any(op in text_lower for op in ["divide", "divided by", "/", "over"]): return "divide"
    if any(op in text_lower for op in ["greater", "larger", "bigger", ">"]): return "greater"
    if any(op in text_lower for op in ["less", "smaller", "<"]): return "less"
    if "even" in text_lower: return "even"
    if "odd" in text_lower: return "odd"
    
    return None

def _solve_sequence(numbers: List[float], text: str) -> Any:
    if len(numbers) < 3: return None

    # 1. Check for Arithmetic Progression (AP)
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers) - 1)]
    if all(math.isclose(d, diffs[0]) for d in diffs):
        return numbers[-1] + diffs[0]

    # 2. Check for Geometric Progression (GP)
    if all(numbers):
        ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers) - 1)]
        if all(math.isclose(r, ratios[0]) for r in ratios):
            return numbers[-1] * ratios[0]
    
    # 3. Check for Simple Quadratic Progression
    if len(diffs) > 2:
        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs) - 1)]
        if all(math.isclose(sd, second_diffs[0]) for sd in second_diffs):
            next_diff = diffs[-1] + second_diffs[0]
            return numbers[-1] + next_diff

    return None


def execute_node(node: str, g: nx.DiGraph, context: Dict[str, Any]) -> Dict[str, Any]:
    kind = g.nodes[node].get("kind")
    node_data = g.nodes[node]
    
    root_text = context.get("input", {}).get("result", "")
    
    inputs: List[Any] = []
    
    if kind != "input":
        for p in g.predecessors(node):
            if p in context and context[p].get("result") is not None:
                if isinstance(context[p]["result"], (list, tuple)):
                    inputs.extend(context[p]["result"])
                else:
                    inputs.append(context[p]["result"])
            
    numbers = _extract_numbers(root_text)
    operation = _detect_operation(root_text)

    # Effective numbers prioritize contextual inputs
    effective_numbers = [i for i in inputs if isinstance(i, (int, float))]
    if not effective_numbers:
        effective_numbers = numbers
    
    if kind == "tool":
        tool_name = node_data.get("tool_type", node) 
        
        # --- math_solve / arithmetic ---
        if tool_name in ["math_solve", "arithmetic"]:
            if "=" in root_text:
                try:
                    left, right = root_text.split("=", 1)
                    res = sp.solve(sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip())))
                    if res:
                        return {"result": res[0] if len(res) == 1 else res, "node": node}
                except Exception:
                    pass
            
            if len(effective_numbers) >= 2 and operation:
                try:
                    a, b = effective_numbers[0], effective_numbers[1]
                    if operation == "add": result = sum(effective_numbers)
                    elif operation == "subtract": result = a - b
                    elif operation == "multiply": result = a * b
                    elif operation == "divide": result = a / b if b != 0 else None
                    else: result = None
                    
                    if result is not None:
                        return {"result": float(result) if result == int(result) else result, "node": node}
                except Exception:
                    pass
            
            try:
                clean_text = root_text.lower().replace("what is", "").replace("calculate", "").replace("compute", "").replace("times", "*").replace("minus", "-").replace("divided by", "/").replace("plus", "+").replace("?", "").strip()
                result = sp.sympify(clean_text)
                if result.is_number:
                    return {"result": float(result), "node": node}
            except Exception:
                pass

        # --- sequence_solver ---
        elif tool_name == "sequence_solver":
            if effective_numbers:
                result = _solve_sequence(effective_numbers, root_text)
                if result is not None:
                    return {"result": result, "node": node}
                
        # --- compare ---
        elif tool_name == "compare":
            if len(effective_numbers) >= 2 and operation:
                a, b = effective_numbers[0], effective_numbers[1]
                if operation == "greater":
                    return {"result": a if a > b else b, "node": node}
                elif operation == "less":
                    return {"result": a if a < b else b, "node": node}
            return {"result": None, "node": node}

        # --- logic_branch ---
        elif tool_name == "logic_branch":
            if inputs and isinstance(inputs[0], bool):
                return {"result": inputs[0], "node": node}
                
            if operation == "even" and effective_numbers:
                return {"result": int(effective_numbers[0]) % 2 == 0, "node": node}
            if operation == "odd" and effective_numbers:
                return {"result": int(effective_numbers[0]) % 2 == 1, "node": node}
            
            if " if " in root_text.lower() and " then " in root_text.lower():
                 return {"result": True, "node": node, "explanation": "Implication pattern detected"}
            
            return {"result": None, "node": node}
        
        # --- fallback ---
        elif tool_name == "fallback":
            if effective_numbers:
                 return {"result": sum(effective_numbers), "node": node}
            return {"result": None, "node": node}

    if kind == "aggregate":
        for p in g.predecessors(node):
            res = context.get(p, {}).get("result")
            if res is not None:
                return {"result": res, "node": node} 
        return {"result": None, "node": node}

    if kind == "verify":
        preds = list(g.predecessors(node))
        res = context.get(preds[0], {}).get("result") if preds else None
        return {"result": res, "node": node}

    if kind == "input":
        return {"result": root_text, "node": node}

    return {"result": None, "node": node}


def execute_graph_topologically(g: nx.DiGraph) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {}
    
    # 1. Execute 'input' node first
    ctx["input"] = execute_node("input", g, ctx)
    
    try:
        topo_order = list(nx.topological_sort(g))
        
        for node in topo_order:
            if node == "input": continue 
            
            ctx[node] = execute_node(node, g, ctx)
            
    except nx.NetworkXError:
        # Fallback to arbitrary execution if cyclic graph occurs (should be rare)
        for node in g.nodes():
            if node == "input": continue
            try:
                ctx[node] = execute_node(node, g, ctx)
            except Exception as e:
                ctx[node] = {"result": None, "node": node, "error": str(e)}
                
    return ctx
