from __future__ import annotations

import copy
import random
import itertools
from dataclasses import dataclass
from typing import Callable, Tuple, Any, Dict, List

import networkx as nx


@dataclass
class EvolutionConfig:
    max_generations: int = 8
    population_size: int = 12
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    
    # NEW: Multi-objective fitness parameters (weights for the Verifier score)
    alpha_confidence: float = 0.85 
    beta_efficiency: float = 0.15 
    complexity_penalty: float = 0.15 # Higher penalty for complexity (was 0.10)


def _random_weight():
    r = random.random()
    return max(0.01, min(1.0, 0.5 + 0.5 * (r - 0.5))) # Ensure weight is between 0.01 and 1.0


def _is_valid_edge(g: nx.DiGraph, src: str, dst: str) -> bool:
    if src == dst:
        return False
    if g.has_edge(src, dst):
        return False
    
    g_temp = g.copy()
    g_temp.add_edge(src, dst)
    try:
        list(nx.topological_sort(g_temp))
        return True
    except nx.NetworkXError:
        return False

# NEW: List of valid tool names for mutation
VALID_TOOLS = ["math_solve", "arithmetic", "logic_branch", "compare", "fallback"]

def mutate_graph(g: nx.DiGraph) -> nx.DiGraph:
    h = copy.deepcopy(g)
    # Added 'swap_tool' for targeted evolution 
    ops = ["weight_jitter", "rewire", "add_skip", "swap_tool"]
    op = random.choice(ops)
    nodes = list(h.nodes())
    if len(nodes) < 2:
        return h
    
    tool_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "tool"]
    
    if op == "weight_jitter":
        for u, v, data in h.edges(data=True):
            data["weight"] = _random_weight()
    
    elif op == "rewire":
        edges = list(h.edges())
        if not edges:
            return h
        src = random.choice(nodes)
        
        for _ in range(5):
            dst = random.choice(nodes)
            if _is_valid_edge(h, src, dst):
                # Remove an existing edge involving src/dst to make way for the new one
                if h.has_edge(src, random.choice(list(h.successors(src)) or [dst])):
                    h.remove_edge(src, random.choice(list(h.successors(src)) or [dst]))
                
                h.add_edge(src, dst, weight=_random_weight())
                break
    
    elif op == "add_skip":
        tool_nodes_list = [n for n, d in h.nodes(data=True) if d.get("kind") == "tool"]
        agg_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "aggregate"]
        
        # Try to create a new connection between any two non-verify/input nodes
        available_nodes = [n for n in nodes if h.nodes[n].get("kind") not in ["input", "verify"]]
        
        if len(available_nodes) >= 2:
            src, dst = random.sample(available_nodes, 2)
            if _is_valid_edge(h, src, dst):
                h.add_edge(src, dst, weight=_random_weight())
    
    elif op == "swap_tool" and tool_nodes:
        node_to_swap = random.choice(tool_nodes)
        old_tool_name = h.nodes[node_to_swap].get("tool_name", "fallback")
        
        # Select a new tool that is different from the old one
        new_tool_name = random.choice([t for t in VALID_TOOLS if t != old_tool_name])
        
        # NOTE: Tool identity is often stored in the node name/ID. 
        # For simplicity, we assume 'kind' is sufficient for a generic swap.
        # In a real system, this would require renaming the node.
        
        h.nodes[node_to_swap]["tool_name"] = new_tool_name # Hypothetical attribute
        h.nodes[node_to_swap]["kind"] = "tool" # Ensure it remains a tool node

    return h


def crossover_graph(g1: nx.DiGraph, g2: nx.DiGraph) -> nx.DiGraph:
    h = nx.DiGraph()
    
    # Node Reconciliation: Inherit all nodes and merge properties
    all_nodes = set(g1.nodes()) | set(g2.nodes())
    for n in all_nodes:
        d1 = g1.nodes.get(n, {})
        d2 = g2.nodes.get(n, {})
        # Merge node attributes, prioritizing g1's 'kind' but mixing other data
        merged_data = {**d2, **d1}
        h.add_node(n, **merged_data)
    
    # Edge Recombination
    edges = set(g1.edges()) | set(g2.edges())
    for u, v in edges:
        if u not in h.nodes or v not in h.nodes:
            continue
            
        if _is_valid_edge(h, u, v):
            w1 = g1.get_edge_data(u, v, {}).get("weight", None)
            w2 = g2.get_edge_data(u, v, {}).get("weight", None)
            
            if w1 is not None and w2 is not None:
                w = 0.5 * w1 + 0.5 * w2 # Average weights
            else:
                w = w1 if w2 is None else w2
                
            if w is None:
                w = _random_weight()
                
            h.add_edge(u, v, weight=w)
            
    return h


def _ensure_valid_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Ensure graph is valid DAG, fix if needed."""
    try:
        list(nx.topological_sort(g))
        return g
    except nx.NetworkXError:
        g_fixed = copy.deepcopy(g)
        
        # Simple cycle breaking: remove edges that create cycles
        edges_to_remove = []
        for u, v in list(g_fixed.edges()):
            if u == v:
                edges_to_remove.append((u, v))
                continue
                
            g_temp = g_fixed.copy()
            g_temp.remove_edge(u, v)
            
            # Check for cycles remaining in the graph without this edge
            is_valid_without_edge = True
            try:
                list(nx.topological_sort(g_temp))
            except nx.NetworkXError:
                is_valid_without_edge = False
            
            # If removing this edge helps fix a cycle, mark it for removal (heuristically)
            if not is_valid_without_edge and nx.find_cycle(g_fixed):
                edges_to_remove.append((u, v))

        # Remove edges with a high weight bias towards simplicity
        for u, v in edges_to_remove:
            if g_fixed.has_edge(u, v):
                g_fixed.remove_edge(u, v)
        
        try:
            list(nx.topological_sort(g_fixed))
            return g_fixed
        except nx.NetworkXError:
            # If still problematic, return a minimal valid graph
            return _create_minimal_graph(g)

def _create_minimal_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Create a minimal valid graph with same nodes."""
    minimal = nx.DiGraph()
    
    for node, data in g.nodes(data=True):
        minimal.add_node(node, **data)
    
    input_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "input"]
    tool_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "tool"]
    agg_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "aggregate"]
    verify_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "verify"]
    
    if input_nodes and tool_nodes:
        for tool in tool_nodes:
            if not minimal.has_edge(input_nodes[0], tool):
                minimal.add_edge(input_nodes[0], tool, weight=0.5)
    
    if tool_nodes and agg_nodes:
        for tool in tool_nodes:
            if not minimal.has_edge(tool, agg_nodes[0]):
                minimal.add_edge(tool, agg_nodes[0], weight=0.5)
    
    if agg_nodes and verify_nodes:
        if not minimal.has_edge(agg_nodes[0], verify_nodes[0]):
            minimal.add_edge(agg_nodes[0], verify_nodes[0], weight=1.0)
    
    return minimal

def evolve_graph(
    base_graph: nx.DiGraph,
    config: EvolutionConfig,
    fitness_fn: Callable[[nx.DiGraph], Tuple[float, Dict[str, Any]]],
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    
    base_graph = _ensure_valid_graph(base_graph)
    
    population = [copy.deepcopy(base_graph)]
    while len(population) < config.population_size:
        mutated = mutate_graph(base_graph)
        mutated = _ensure_valid_graph(mutated)
        population.append(mutated)

    best_graph = base_graph
    best_score = float("-inf")
    best_meta: Dict[str, Any] = {"trace": []}

    for gen in range(config.max_generations):
        scored = []
        for g in population:
            try:
                g = _ensure_valid_graph(g)
                # The fitness_fn now returns a highly calibrated score (from verifier.py)
                raw_score, meta = fitness_fn(g) 
                
                # Apply Multi-Objective Fitness: prioritize confidence but penalize complexity 
                # Complexity score is already inside the score from verifier.py (efficiency)
                final_fitness_score = raw_score 

                scored.append((final_fitness_score, g, meta))
            except Exception as e:
                scored.append((0.0, g, {"error": str(e)}))
        
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score, best_graph, best_meta = scored[0]
        best_meta.setdefault("trace", []).append({"generation": gen, "best_score": best_score, "max_pop_score": scored[0][0]})

        # Elitism: Keep the top 25% of the population
        elites = [g for _, g, _ in scored[: max(2, len(scored) // 4)]]
        new_pop = elites.copy()
        
        while len(new_pop) < config.population_size:
            try:
                # Selection (using Elitism)
                if random.random() < config.crossover_rate and len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                    child = crossover_graph(p1, p2)
                else:
                    child = copy.deepcopy(random.choice(elites))
                
                # Mutation
                if random.random() < config.mutation_rate:
                    child = mutate_graph(child)
                
                child = _ensure_valid_graph(child)
                new_pop.append(child)
            except Exception:
                # Fail-safe: if child creation fails, copy an elite
                new_pop.append(copy.deepcopy(random.choice(elites)))
        
        population = new_pop

    return _ensure_valid_graph(best_graph), best_meta