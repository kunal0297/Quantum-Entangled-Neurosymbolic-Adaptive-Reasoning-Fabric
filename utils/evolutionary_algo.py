from __future__ import annotations

from typing import Callable, Tuple, Any, Dict
import networkx as nx
from src.graph_evolver import EvolutionConfig, evolve_graph


def run_evolution(
    graph: nx.DiGraph,
    fitness_fn: Callable[[nx.DiGraph], Tuple[float, Dict[str, Any]]],
    max_generations: int = 8,
    population_size: int = 12,
) -> Tuple[nx.DiGraph, Dict[str, Any]]:
    cfg = EvolutionConfig(max_generations=max_generations, population_size=population_size)
    return evolve_graph(graph, cfg, fitness_fn)


