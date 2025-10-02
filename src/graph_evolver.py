from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, Tuple, Any, Dict

import networkx as nx


@dataclass
class EvolutionConfig:
    max_generations: int = 8
    population_size: int = 12
    mutation_rate: float = 0.3
    crossover_rate: float = 0.6
    alpha_accuracy: float = 0.8
    beta_efficiency: float = 0.2


def _random_weight():
    r = random.random()
    return 0.5 + 0.5 * (r - 0.5)


def _is_valid_edge(g: nx.DiGraph, src: str, dst: str) -> bool:
    """Check if adding an edge would create a cycle or invalid structure."""
    if src == dst:
        return False
    if g.has_edge(src, dst):
        return False
    
    # Test if adding this edge would create a cycle
    g_temp = g.copy()
    g_temp.add_edge(src, dst)
    try:
        list(nx.topological_sort(g_temp))
        return True
    except nx.NetworkXError:
        return False

def mutate_graph(g: nx.DiGraph) -> nx.DiGraph:
    h = copy.deepcopy(g)
    ops = ["weight_jitter", "rewire", "add_skip"]  # Removed toggle_edge to prevent cycles
    op = random.choice(ops)
    nodes = list(h.nodes())
    if len(nodes) < 2:
        return h
    
    if op == "weight_jitter":
        for u, v, data in h.edges(data=True):
            data["weight"] = max(0.0, min(1.0, data.get("weight", 0.5) + random.uniform(-0.1, 0.1)))
    
    elif op == "rewire":
        edges = list(h.edges())
        if not edges:
            return h
        e = random.choice(edges)
        src = e[0]
        # Try to find a valid destination
        for _ in range(5):  # Maximum 5 attempts
            dst = random.choice(nodes)
            if _is_valid_edge(h, src, dst):
                h.remove_edge(*e)
                h.add_edge(src, dst, weight=_random_weight())
                break
    
    elif op == "add_skip":
        # Only add forward connections to maintain DAG structure
        input_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "input"]
        tool_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "tool"]
        agg_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "aggregate"]
        
        if input_nodes and tool_nodes:
            src = random.choice(input_nodes)
            dst = random.choice(tool_nodes)
            if _is_valid_edge(h, src, dst):
                h.add_edge(src, dst, weight=_random_weight())
        elif tool_nodes and agg_nodes:
            src = random.choice(tool_nodes)
            dst = random.choice(agg_nodes)
            if _is_valid_edge(h, src, dst):
                h.add_edge(src, dst, weight=_random_weight())
    
    return h


def crossover_graph(g1: nx.DiGraph, g2: nx.DiGraph) -> nx.DiGraph:
    h = nx.DiGraph()
    for n, d in g1.nodes(data=True):
        h.add_node(n, **d)
    for n, d in g2.nodes(data=True):
        if not h.has_node(n):
            h.add_node(n, **d)
    
    edges = set(list(g1.edges()) + list(g2.edges()))
    for u, v in edges:
        # Only add edge if it doesn't create a cycle
        if _is_valid_edge(h, u, v):
            w1 = g1.get_edge_data(u, v, {}).get("weight", None)
            w2 = g2.get_edge_data(u, v, {}).get("weight", None)
            if w1 is not None and w2 is not None:
                w = 0.5 * (w1 + w2)
            else:
                w = w1 if w2 is None else w2
            if w is None:
                w = _random_weight()
            h.add_edge(u, v, weight=w)
    
    return h


def _ensure_valid_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Ensure graph is valid DAG, fix if needed."""
    try:
        # Check if it's a valid DAG
        list(nx.topological_sort(g))
        return g
    except nx.NetworkXError:
        # Graph has cycles, fix by removing problematic edges
        g_fixed = copy.deepcopy(g)
        
        # Simple cycle breaking: remove edges that create cycles
        edges_to_remove = []
        for u, v in list(g_fixed.edges()):
            g_temp = g_fixed.copy()
            g_temp.remove_edge(u, v)
            try:
                list(nx.topological_sort(g_temp))
                # Edge is not necessary for maintaining structure
                edges_to_remove.append((u, v))
            except nx.NetworkXError:
                continue
        
        # Remove some problematic edges
        for u, v in edges_to_remove[:len(edges_to_remove)//2]:
            if g_fixed.has_edge(u, v):
                g_fixed.remove_edge(u, v)
        
        # Final check
        try:
            list(nx.topological_sort(g_fixed))
            return g_fixed
        except nx.NetworkXError:
            # If still problematic, return a minimal valid graph
            return _create_minimal_graph(g)

def _create_minimal_graph(g: nx.DiGraph) -> nx.DiGraph:
    """Create a minimal valid graph with same nodes."""
    minimal = nx.DiGraph()
    
    # Add all nodes
    for node, data in g.nodes(data=True):
        minimal.add_node(node, **data)
    
    # Add only essential edges in proper order
    input_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "input"]
    tool_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "tool"]
    agg_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "aggregate"]
    verify_nodes = [n for n, d in minimal.nodes(data=True) if d.get("kind") == "verify"]
    
    # Connect in proper order: input -> tools -> aggregate -> verify
    if input_nodes and tool_nodes:
        for tool in tool_nodes:
            minimal.add_edge(input_nodes[0], tool, weight=0.5)
    
    if tool_nodes and agg_nodes:
        for tool in tool_nodes:
            minimal.add_edge(tool, agg_nodes[0], weight=0.5)
    
    if agg_nodes and verify_nodes:
        minimal.add_edge(agg_nodes[0], verify_nodes[0], weight=1.0)
    
    return minimal

def evolve_graph(
    base_graph: nx.DiGraph,
    config: EvolutionConfig,
    fitness_fn: Callable[[nx.DiGraph], Tuple[float, Dict[str, Any]]],
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    
    # Ensure base graph is valid
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
                # Ensure graph is valid before fitness evaluation
                g = _ensure_valid_graph(g)
                score, meta = fitness_fn(g)
                scored.append((score, g, meta))
            except Exception as e:
                # If fitness evaluation fails, give low score
                scored.append((0.0, g, {"error": str(e)}))
        
        scored.sort(key=lambda x: x[0], reverse=True)

        if scored[0][0] > best_score:
            best_score, best_graph, best_meta = scored[0]
        best_meta.setdefault("trace", []).append({"generation": gen, "best_score": best_score})

        elites = [g for _, g, _ in scored[: max(2, len(scored) // 4)]]
        new_pop = elites.copy()
        
        while len(new_pop) < config.population_size:
            try:
                if random.random() < config.crossover_rate and len(elites) >= 2:
                    p1, p2 = random.sample(elites, 2)
                    child = crossover_graph(p1, p2)
                else:
                    child = copy.deepcopy(random.choice(elites))
                
                if random.random() < config.mutation_rate:
                    child = mutate_graph(child)
                
                child = _ensure_valid_graph(child)
                new_pop.append(child)
            except Exception:
                # If child creation fails, just copy an elite
                new_pop.append(copy.deepcopy(random.choice(elites)))
        
        population = new_pop

    return _ensure_valid_graph(best_graph), best_meta


