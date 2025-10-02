from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import hashlib

import networkx as nx
import numpy as np

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    torch = None


@dataclass
class AnalogySample:
    question: str
    graph: Dict  # Serialized graph
    solution: str
    performance_score: float
    structure_hash: str


class AnalogyDetector:
    """Detects analogous problems and reuses successful graph structures."""
    
    def __init__(self, memory_file: str = "data/analogy_memory.json", max_samples: int = 500):
        self.memory_file = memory_file
        self.max_samples = max_samples
        self.samples: List[AnalogySample] = []
        
        # Optional tiny embedding model
        self.tokenizer = None
        self.embed_model = None
        if AutoTokenizer and AutoModel:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model.eval()
            except Exception:
                pass
        
        self._load_memory()
    
    def _compute_structure_hash(self, graph: nx.DiGraph) -> str:
        """Compute a hash representing graph structure."""
        # Create a canonical representation
        node_kinds = sorted([graph.nodes[n].get("kind", "unknown") for n in graph.nodes()])
        edge_patterns = []
        
        for u, v, d in graph.edges(data=True):
            u_kind = graph.nodes[u].get("kind", "unknown")
            v_kind = graph.nodes[v].get("kind", "unknown")
            edge_patterns.append(f"{u_kind}->{v_kind}")
        
        edge_patterns.sort()
        
        structure_str = f"nodes:{','.join(node_kinds)};edges:{','.join(edge_patterns)}"
        return hashlib.md5(structure_str.encode()).hexdigest()[:12]
    
    def _extract_semantic_features(self, question: str) -> Optional[np.ndarray]:
        """Extract semantic features using tiny embedding model."""
        if not self.tokenizer or not self.embed_model:
            return None
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=64)
                outputs = self.embed_model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy().flatten()
        except Exception:
            return None
    
    def _compute_text_similarity(self, q1: str, q2: str) -> float:
        """Compute similarity between two questions."""
        # Semantic similarity if embeddings available
        if self.embed_model:
            emb1 = self._extract_semantic_features(q1)
            emb2 = self._extract_semantic_features(q2)
            if emb1 is not None and emb2 is not None:
                # Cosine similarity
                dot_product = np.dot(emb1, emb2)
                norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
                if norm_product > 0:
                    return dot_product / norm_product
        
        # Fallback: simple lexical overlap
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def find_analogous_graphs(self, question: str, min_similarity: float = 0.4) -> List[Tuple[float, nx.DiGraph]]:
        """Find analogous problem graphs based on question similarity."""
        if not self.samples:
            return []
        
        candidates = []
        
        for sample in self.samples:
            # Compute similarity
            text_sim = self._compute_text_similarity(question, sample.question)
            
            if text_sim >= min_similarity:
                # Reconstruct graph from serialized form
                try:
                    graph = nx.node_link_graph(sample.graph)
                    # Weight by both similarity and past performance
                    combined_score = text_sim * 0.7 + sample.performance_score * 0.3
                    candidates.append((combined_score, graph))
                except Exception:
                    continue  # Skip malformed graphs
        
        # Sort by combined score, return top candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:3]  # Top 3 analogous graphs
    
    def find_structural_analogs(self, graph: nx.DiGraph) -> List[Tuple[float, nx.DiGraph]]:
        """Find graphs with similar structure."""
        target_hash = self._compute_structure_hash(graph)
        
        candidates = []
        for sample in self.samples:
            if sample.structure_hash == target_hash:
                try:
                    analog_graph = nx.node_link_graph(sample.graph)
                    candidates.append((sample.performance_score, analog_graph))
                except Exception:
                    continue
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:2]  # Top 2 structural analogs
    
    def create_hybrid_graph(self, base_graph: nx.DiGraph, analogs: List[Tuple[float, nx.DiGraph]]) -> nx.DiGraph:
        """Create a hybrid graph combining base with analogous structures."""
        if not analogs:
            return base_graph
        
        # Start with base graph
        hybrid = base_graph.copy()
        
        # Add beneficial edges from analogs
        for score, analog in analogs:
            if score < 0.3:  # Don't use low-quality analogs
                continue
            
            for u, v, d in analog.edges(data=True):
                # Only add edge if both nodes exist in hybrid and edge doesn't exist
                if hybrid.has_node(u) and hybrid.has_node(v) and not hybrid.has_edge(u, v):
                    # Weight by analog quality
                    weight = d.get("weight", 0.5) * score
                    hybrid.add_edge(u, v, weight=weight, source="analogy")
        
        return hybrid
    
    def store_successful_solution(self, question: str, graph: nx.DiGraph, solution: str, performance_score: float):
        """Store a successful solution for future analogical reasoning."""
        if performance_score < 0.5:  # Only store reasonably good solutions
            return
        
        # Serialize graph
        try:
            graph_data = nx.node_link_data(graph)
            structure_hash = self._compute_structure_hash(graph)
            
            sample = AnalogySample(
                question=question,
                graph=graph_data,
                solution=solution,
                performance_score=performance_score,
                structure_hash=structure_hash
            )
            
            self.samples.append(sample)
            
            # Trim if too many samples
            if len(self.samples) > self.max_samples:
                # Keep highest-performing samples
                self.samples.sort(key=lambda s: s.performance_score, reverse=True)
                self.samples = self.samples[:self.max_samples]
            
            self._save_memory()
            
        except Exception:
            pass  # Fail gracefully
    
    def _save_memory(self):
        """Save analogy memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            data = [asdict(sample) for sample in self.samples]
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def _load_memory(self):
        """Load analogy memory from disk."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                samples = []
                for item in data:
                    try:
                        sample = AnalogySample(**item)
                        samples.append(sample)
                    except Exception:
                        continue
                
                self.samples = samples
        except Exception:
            self.samples = []
