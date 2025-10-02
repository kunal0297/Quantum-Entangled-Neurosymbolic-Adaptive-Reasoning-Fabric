from __future__ import annotations

import random
import math
from typing import Dict, List, Tuple, Any
import numpy as np
import networkx as nx


class UncertaintyModeler:
    """Models uncertainty for creative exploration in reasoning paths."""
    
    def __init__(self, exploration_rate: float = 0.15, quantum_seed: bool = True):
        self.exploration_rate = exploration_rate
        self.uncertainty_cache: Dict[str, float] = {}
        
        # Quantum-inspired randomness initialization
        if quantum_seed:
            self._init_quantum_random()
    
    def _init_quantum_random(self):
        """Initialize quantum-inspired random number generation."""
        # Use time and mathematical constants for quantum-like randomness
        import time
        seed = int((time.time() * 1000) % (2**32))
        random.seed(seed)
        np.random.seed(seed % (2**31))
    
    def estimate_node_uncertainty(self, node: str, graph: nx.DiGraph, context: Dict[str, Any]) -> float:
        """Estimate uncertainty for a specific node's execution."""
        cache_key = f"{node}_{len(graph.nodes())}"
        
        if cache_key in self.uncertainty_cache:
            return self.uncertainty_cache[cache_key]
        
        uncertainty = 0.5  # Base uncertainty
        
        # Factor 1: Node type uncertainty
        node_kind = graph.nodes[node].get("kind", "unknown")
        kind_uncertainties = {
            "input": 0.1,      # Input is certain
            "verify": 0.2,     # Verification is fairly certain
            "aggregate": 0.3,  # Aggregation has some uncertainty
            "tool": 0.6,       # Tool execution has higher uncertainty
            "unknown": 0.8     # Unknown nodes are very uncertain
        }
        uncertainty = kind_uncertainties.get(node_kind, 0.5)
        
        # Factor 2: Connectivity uncertainty
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        
        # Nodes with many connections are potentially more uncertain
        connectivity_uncertainty = min(0.3, (in_degree + out_degree) * 0.05)
        uncertainty += connectivity_uncertainty
        
        # Factor 3: Edge weight variance
        incoming_weights = [graph[pred][node].get("weight", 0.5) 
                          for pred in graph.predecessors(node)]
        if incoming_weights:
            weight_variance = np.var(incoming_weights)
            uncertainty += weight_variance * 0.5
        
        # Factor 4: Historical performance (if available)
        if node in context:
            result = context[node].get("result")
            if result is None:
                uncertainty += 0.3  # Failed execution increases uncertainty
            elif isinstance(result, (list, tuple)) and len(result) > 1:
                uncertainty += 0.1  # Multiple solutions indicate uncertainty
        
        # Clamp uncertainty to [0, 1]
        uncertainty = max(0.0, min(1.0, uncertainty))
        
        # Cache for efficiency
        self.uncertainty_cache[cache_key] = uncertainty
        return uncertainty
    
    def should_explore_alternative(self, node: str, graph: nx.DiGraph, context: Dict[str, Any]) -> bool:
        """Decide whether to explore alternative reasoning path for a node."""
        uncertainty = self.estimate_node_uncertainty(node, graph, context)
        
        # Quantum-inspired decision: use uncertainty as probability
        exploration_threshold = self.exploration_rate + uncertainty * 0.3
        
        # Add some quantum randomness
        quantum_factor = math.sin(random.random() * math.pi) ** 2
        final_threshold = exploration_threshold * quantum_factor
        
        return random.random() < final_threshold
    
    def generate_alternative_paths(self, graph: nx.DiGraph, uncertain_nodes: List[str]) -> List[nx.DiGraph]:
        """Generate alternative graph variants for uncertain nodes."""
        alternatives = []
        
        for _ in range(min(3, len(uncertain_nodes))):  # Generate up to 3 alternatives
            alt_graph = graph.copy()
            
            # Randomly select some uncertain nodes to modify
            nodes_to_modify = random.sample(uncertain_nodes, 
                                          min(2, len(uncertain_nodes)))
            
            for node in nodes_to_modify:
                self._apply_uncertainty_mutation(alt_graph, node)
            
            alternatives.append(alt_graph)
        
        return alternatives
    
    def _apply_uncertainty_mutation(self, graph: nx.DiGraph, node: str):
        """Apply uncertainty-based mutations to a node."""
        mutation_type = random.choice([
            "weight_noise",
            "alternative_connection",
            "creative_skip",
            "tool_substitution"
        ])
        
        if mutation_type == "weight_noise":
            # Add noise to edge weights
            for pred in list(graph.predecessors(node)):
                if graph.has_edge(pred, node):
                    current_weight = graph[pred][node].get("weight", 0.5)
                    noise = random.gauss(0, 0.1)  # Small Gaussian noise
                    new_weight = max(0.1, min(1.0, current_weight + noise))
                    graph[pred][node]["weight"] = new_weight
        
        elif mutation_type == "alternative_connection":
            # Try connecting to alternative nodes
            possible_targets = [n for n in graph.nodes() 
                              if n != node and not graph.has_edge(node, n)]
            if possible_targets:
                target = random.choice(possible_targets)
                graph.add_edge(node, target, weight=random.uniform(0.2, 0.8),
                             source="uncertainty_exploration")
        
        elif mutation_type == "creative_skip":
            # Add creative skip connections
            if graph.has_node("input") and not graph.has_edge("input", node):
                # Small probability of direct skip
                if random.random() < 0.3:
                    graph.add_edge("input", node, weight=random.uniform(0.1, 0.4),
                                 source="creative_skip")
        
        elif mutation_type == "tool_substitution":
            # If this is a tool node, consider alternative tools
            if graph.nodes[node].get("kind") == "tool":
                alternative_tools = ["math_solve", "arithmetic", "logic_branch", "compare"]
                current_tool = node
                if current_tool in alternative_tools:
                    alternatives = [t for t in alternative_tools if t != current_tool]
                    if alternatives and random.random() < 0.4:
                        # Create alternative tool node
                        alt_tool = random.choice(alternatives)
                        if not graph.has_node(alt_tool):
                            graph.add_node(alt_tool, kind="tool", source="uncertainty_substitute")
                            # Copy some connections
                            for pred in list(graph.predecessors(node)):
                                if random.random() < 0.5:  # 50% chance to copy edge
                                    weight = graph[pred][node].get("weight", 0.5) * 0.8
                                    graph.add_edge(pred, alt_tool, weight=weight)
    
    def compute_exploration_bonus(self, graph: nx.DiGraph, context: Dict[str, Any]) -> float:
        """Compute bonus score for creative exploration."""
        bonus = 0.0
        
        # Count creative elements added by uncertainty exploration
        creative_edges = sum(1 for u, v, d in graph.edges(data=True)
                           if d.get("source") in ["uncertainty_exploration", "creative_skip"])
        
        creative_nodes = sum(1 for n, d in graph.nodes(data=True)
                           if d.get("source") == "uncertainty_substitute")
        
        # Small bonus for creativity, but don't override correctness
        bonus += creative_edges * 0.02
        bonus += creative_nodes * 0.05
        
        # Bonus for successful uncertain explorations
        uncertain_successes = 0
        for node in graph.nodes():
            if node in context:
                result = context[node].get("result")
                uncertainty = self.estimate_node_uncertainty(node, graph, context)
                if result is not None and uncertainty > 0.6:
                    uncertain_successes += 1
        
        bonus += uncertain_successes * 0.1
        
        return min(0.3, bonus)  # Cap exploration bonus
    
    def get_uncertainty_report(self, graph: nx.DiGraph, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate uncertainty analysis report."""
        uncertainties = {}
        for node in graph.nodes():
            uncertainties[node] = self.estimate_node_uncertainty(node, graph, context)
        
        avg_uncertainty = np.mean(list(uncertainties.values()))
        high_uncertainty_nodes = [node for node, unc in uncertainties.items() if unc > 0.7]
        
        return {
            "node_uncertainties": uncertainties,
            "average_uncertainty": avg_uncertainty,
            "high_uncertainty_nodes": high_uncertainty_nodes,
            "exploration_rate": self.exploration_rate,
            "creative_elements": sum(1 for u, v, d in graph.edges(data=True)
                                   if "uncertainty" in d.get("source", ""))
        }
    
    def adaptive_exploration_rate(self, success_rate: float):
        """Adaptively adjust exploration rate based on recent success."""
        if success_rate > 0.8:
            # High success rate - increase exploration
            self.exploration_rate = min(0.3, self.exploration_rate * 1.1)
        elif success_rate < 0.5:
            # Low success rate - decrease exploration
            self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
        
        # Add some quantum randomness to prevent getting stuck
        quantum_adjustment = (random.random() - 0.5) * 0.02
        self.exploration_rate = max(0.05, min(0.3, self.exploration_rate + quantum_adjustment))
