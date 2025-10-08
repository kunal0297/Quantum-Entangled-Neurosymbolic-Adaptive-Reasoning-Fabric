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
    mutation_rate: float = 0.40 # Slightly increased to encourage dynamic changes
    crossover_rate: float = 0.6
    
    # Fitness parameters (Optimized for balance and confidence)
    alpha_confidence: float = 0.85 
    beta_efficiency: float = 0.15 
    complexity_penalty: float = 0.10 # Maintained balance for complex problems


def _random_weight():
    r = random.random()
    return max(0.01, min(1.0, 0.5 + 0.5 * (r - 0.5))) 


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

# Includes the new tool 'sequence_solver'
VALID_TOOLS = ["math_solve", "arithmetic", "logic_branch", "compare", "sequence_solver", "fallback"]

def _get_new_node_id(g: nx.DiGraph, base_name: str) -> str:
    """Generates a unique ID for a new node based on its base name."""
    i = 1
    while f"{base_name}_{i}" in g.nodes:
        i += 1
    return f"{base_name}_{i}"


def mutate_graph(g: nx.DiGraph) -> nx.DiGraph:
    h = copy.deepcopy(g)
    
    # Added core dynamic operations: 'add_node' and 'remove_node'
    ops = ["weight_jitter", "rewire", "add_skip", "swap_tool", "add_node", "remove_node"]
    op = random.choice(ops)
    nodes = list(h.nodes())
    tool_nodes = [n for n, d in h.nodes(data=True) if d.get("kind") == "tool"]
    non_fixed_nodes = [n for n in nodes if h.nodes[n].get("kind") not in ["input", "aggregate", "verify"]]


    if op == "weight_jitter":
        for u, v, data in h.edges(data=True):
            data["weight"] = _random_weight()
    
    elif op == "rewire":
        edges = list(h.edges())
        if not edges: return h
        u_remove, v_remove = random.choice(edges)
        h.remove_edge(u_remove, v_remove)
        for _ in range(5):
            src = random.choice(nodes)
            dst = random.choice(nodes)
            if _is_valid_edge(h, src, dst):
                h.add_edge(src, dst, weight=_random_weight())
                break
    
    elif op == "add_skip":
        available_nodes = [n for n in nodes if h.nodes[n].get("kind") not in ["verify"]]
        if len(available_nodes) >= 2:
            src, dst = random.sample(available_nodes, 2)
            if _is_valid_edge(h, src, dst):
                h.add_edge(src, dst, weight=_random_weight())
    
    elif op == "swap_tool" and tool_nodes:
        node_to_swap = random.choice(tool_nodes)
        current_tool_name = h.nodes[node_to_swap].get("tool_name", node_to_swap)
        new_tool_name = random.choice([t for t in VALID_TOOLS if t != current_tool_name])
        h.nodes[node_to_swap]["tool_name"] = new_tool_name 
        h.nodes[node_to_swap]["tool_type"] = new_tool_name # Added tool_type attribute for clarity

    # ==========================================================
    # CORE DYNAMIC GRAPH EVOLUTION OPERATIONS
    # ==========================================================
    elif op == "add_node":
        if random.random() < 0.5:
            # Add a node between two existing connected nodes
            source_nodes = [n for n in nodes if n not in ["verify"]]
            if not source_nodes: return h
            src = random.choice(source_nodes)
            if list(h.successors(src)):
                dst = random.choice(list(h.successors(src)))
                
                new_tool_type = random.choice(VALID_TOOLS)
                new_node_id = _get_new_node_id(h, new_tool_type)

                # Insert node: src -> new_node -> dst
                h.remove_edge(src, dst)
                h.add_node(new_node_id, kind="tool", tool_type=new_tool_type)
                h.add_edge(src, new_node_id, weight=_random_weight())
                h.add_edge(new_node_id, dst, weight=_random_weight())
        else:
            # Add a node connected from 'input' to 'aggregate'
            new_tool_type = random.choice(VALID_TOOLS)
            new_node_id = _get_new_node_id(h, new_tool_type)
            h.add_node(new_node_id, kind="tool", tool_type=new_tool_type)
            h.add_edge("input", new_node_id, weight=_random_weight())
            h.add_edge(new_node_id, "aggregate", weight=_random_weight())
    
    elif op == "remove_node" and non_fixed_nodes:
        node_to_remove = random.choice(non_fixed_nodes)
        
        predecessors = list(h.predecessors(node_to_remove))
        successors = list(h.successors(node_to_remove))
        
        h.remove_node(node_to_remove)
        
        # Reconnect predecessors to successors
        for pred in predecessors:
            for succ in successors:
                if _is_valid_edge(h, pred, succ):
                    h.add_edge(pred, succ, weight=_random_weight())
        
    return h
# ... [rest of the file content remains the same: crossover_graph, _ensure_valid_graph, _create_minimal_graph, evolve_graph]