from __future__ import annotations

from typing import Any, Dict, Tuple, List
import networkx as nx
import sympy as sp
import re
import math


def _extract_numbers(text: str):
    """Extract all numbers from text."""
    return [float(x) for x in re.findall(r'-?\d+\.?\d*', text)]

def _detect_operation(text: str):
    """Detect mathematical operation from text."""
    text_lower = text.lower()
    
    if any(op in text_lower for op in ["plus", "add", "+", "sum"]):
        return "add"
    if any(op in text_lower for op in ["minus", "subtract", "-", "difference"]):
        return "subtract"
    if any(op in text_lower for op in ["times", "multiply", "*", "product"]):
        return "multiply"
    if any(op in text_lower for op in ["divide", "divided by", "/", "over"]):
        return "divide"
    if any(op in text_lower for op in ["greater", "larger", "bigger", ">"]):
        return "greater"
    if any(op in text_lower for op in ["less", "smaller", "<"]):
        return "less"
    if "even" in text_lower:
        return "even"
    if "odd" in text_lower:
        return "odd"
    
    return None

def _solve_sequence(numbers: List[float], text: str) -> Any:
    """
    Implements the specialized logic for the 'sequence_solver' tool.
    Tries to find simple patterns (Arithmetic, Geometric) and return the next term.
    """
    if len(numbers) < 3:
        return None # Need at least 3 terms to reliably detect a pattern

    # 1. Check for Arithmetic Progression (AP)
    diffs = [numbers[i+1] - numbers[i] for i in range(len(numbers) - 1)]
    if all(math.isclose(d, diffs[0]) for d in diffs):
        return numbers[-1] + diffs[0] # Next number is last + common difference

    # 2. Check for Geometric Progression (GP)
    if all(numbers): # Ensure no division by zero
        ratios = [numbers[i+1] / numbers[i] for i in range(len(numbers) - 1)]
        if all(math.isclose(r, ratios[0]) for r in ratios):
            return numbers[-1] * ratios[0] # Next number is last * common ratio
    
    # 3. Check for Simple Quadratic Progression (Second difference is constant)
    if len(diffs) > 2:
        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs) - 1)]
        if all(math.isclose(sd, second_diffs[0]) for sd in second_diffs):
            # Calculate next difference, then next number
            next_diff = diffs[-1] + second_diffs[0]
            return numbers[-1] + next_diff

    return None


def execute_node(node: str, g: nx.DiGraph, context: Dict[str, Any]) -> Dict[str, Any]:
    kind = g.nodes[node].get("kind")
    node_data = g.nodes[node]
    
    # --- Context-Aware Data Retrieval (CRITICAL FOR MULTI-STEP REASONING) ---
    
    # If a node has predecessors, pull their results into 'inputs'
    inputs: List[Any] = []
    
    # If the node is not 'input' itself, look up results from all predecessors
    if kind != "input":
        for p in g.predecessors(node):
            if p in context and context[p].get("result") is not None:
                # If predecessor result is a list, extend inputs with individual elements
                if isinstance(context[p]["result"], (list, tuple)):
                    inputs.extend(context[p]["result"])
                else:
                    # Otherwise, append the single result
                    inputs.append(context[p]["result"])
            
    # For initial tools, we still rely on the root text if no context inputs are available
    root_text = context.get("input", {}).get("result", "")
    
    # Extract raw numbers and operation from the root text as a fallback/initial hint
    numbers = _extract_numbers(root_text)
    operation = _detect_operation(root_text)

    # --- Tool Execution Logic ---

    if kind == "tool":
        # Check if the node is a specific named tool, or if the node ID implies the tool (fallback for old graphs)
        tool_name = node_data.get("tool_type", node) 
        
        # Combine inputs and numbers from text for flexibility. Prioritize contextual inputs.
        effective_numbers = [i for i in inputs if isinstance(i, (int, float))]
        if not effective_numbers:
            effective_numbers = numbers
        
        # --- Tool 1 & 2: math_solve / arithmetic ---
        if tool_name in ["math_solve", "arithmetic"]:
            # Logic for Equations (SymPy Solve)
            if "=" in root_text:
                try:
                    left, right = root_text.split("=", 1)
                    res = sp.solve(sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip())))
                    if res:
                        return {"result": res[0] if len(res) == 1 else res, "node": node}
                except Exception:
                    pass
            
            # Logic for Arithmetic (using contextual/extracted numbers)
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
            
            # Fallback: Try SymPy direct evaluation on cleaned text
            try:
                clean_text = root_text.lower().replace("what is", "").replace("calculate", "").replace("compute", "").replace("times", "*").replace("minus", "-").replace("divided by", "/").replace("plus", "+").replace("?", "").strip()
                result = sp.sympify(clean_text)
                if result.is_number:
                    return {"result": float(result), "node": node}
            except Exception:
                pass

        # --- Tool 3: sequence_solver (NEW SPECIALIZED TOOL) ---
        elif tool_name == "sequence_solver":
            # Use numbers extracted from the root text OR from predecessors
            if effective_numbers:
                result = _solve_sequence(effective_numbers, root_text)
                if result is not None:
                    return {"result": result, "node": node}
                
        # --- Tool 4: compare ---
        elif tool_name == "compare":
            if len(effective_numbers) >= 2 and operation:
                a, b = effective_numbers[0], effective_numbers[1]
                if operation == "greater":
                    return {"result": a if a > b else b, "node": node}
                elif operation == "less":
                    return {"result": a if a < b else b, "node": node}
            return {"result": None, "node": node}

        # --- Tool 5: logic_branch ---
        elif tool_name == "logic_branch":
            # Contextual Logic: If predecessor result is a Boolean
            if inputs and isinstance(inputs[0], bool):
                return {"result": inputs[0], "node": node}
                
            # Logic on extracted text/numbers (Fallback)
            if operation == "even" and effective_numbers:
                return {"result": int(effective_numbers[0]) % 2 == 0, "node": node}
            if operation == "odd" and effective_numbers:
                return {"result": int(effective_numbers[0]) % 2 == 1, "node": node}
            
            # Basic Implication Detection
            if " if " in root_text.lower() and " then " in root_text.lower():
                 return {"result": True, "node": node, "explanation": "Implication pattern detected"}
            
            return {"result": None, "node": node}
        
        # --- Tool 6: fallback ---
        elif tool_name == "fallback":
            # Simplistic aggregation/summation as a last resort
            if effective_numbers:
                 return {"result": sum(effective_numbers), "node": node}
            return {"result": None, "node": node}


    if kind == "aggregate":
        # The aggregate node now correctly pulls the result from any successful predecessor.
        for p in g.predecessors(node):
            res = context.get(p, {}).get("result")
            if res is not None:
                # Prioritize the result from the path with the highest edge weight (confidence)
                # or just return the first valid result found in topological order (simpler)
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
    
    # 1. Execute 'input' node first to populate root_text
    ctx["input"] = execute_node("input", g, ctx)
    
    try:
        topo_order = list(nx.topological_sort(g))
        
        # Execute nodes in order, skipping 'input' as it's done
        for node in topo_order:
            if node == "input": continue 
            
            # Note: The context check is crucial. If a predecessor node failed (result=None),
            # the current node's execution logic must handle the lack of inputs.
            ctx[node] = execute_node(node, g, ctx)
            
    except nx.NetworkXError:
        # Should be rare due to _ensure_valid_graph in evolver
        print("Warning: Graph contains cycles, execution may be unstable.")
        # Fallback to arbitrary execution
        for node in g.nodes():
            if node == "input": continue
            try:
                ctx[node] = execute_node(node, g, ctx)
            except Exception as e:
                ctx[node] = {"result": None, "node": node, "error": str(e)}
                
    return ctx