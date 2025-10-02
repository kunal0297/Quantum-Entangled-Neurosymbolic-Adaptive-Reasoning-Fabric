from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import hashlib

import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None


@dataclass
class ProblemFeatures:
    length: int
    has_math: bool
    has_logic: bool
    has_comparison: bool
    keyword_hash: str
    complexity_score: float


@dataclass
class GraphMetrics:
    nodes: int
    edges: int
    density: float
    avg_weight: float
    final_score: float
    evolution_generations: int


@dataclass
class MetaLearningRecord:
    problem_features: ProblemFeatures
    graph_metrics: GraphMetrics
    config_used: Dict[str, float]
    success: bool


class TinyMetaNet(nn.Module):
    """Tiny neural network to predict optimal evolution config."""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(x))  # Output [0,1] for config params


class MetaLearner:
    """Meta-learning component to optimize graph evolution based on problem history."""
    
    def __init__(self, memory_file: str = "data/meta_memory.json", max_records: int = 1000):
        self.memory_file = memory_file
        self.max_records = max_records
        self.records: List[MetaLearningRecord] = []
        self.meta_net = None
        
        if torch is not None:
            self.meta_net = TinyMetaNet()
            self.meta_net.eval()
        
        self._load_memory()
    
    def _extract_features(self, question: str) -> ProblemFeatures:
        """Extract features from question for meta-learning."""
        import re
        
        # Basic features
        length = len(question.split())
        has_math = bool(re.search(r"(\d+|solve|equation|sum|product)", question, re.I))
        has_logic = bool(re.search(r"(if|then|all|some|none|true|false)", question, re.I))
        has_comparison = bool(re.search(r"(greater|less|more|fewer|compare)", question, re.I))
        
        # Keyword hash for similarity
        keywords = re.findall(r"[a-zA-Z]+", question.lower())
        keyword_hash = hashlib.md5("".join(sorted(keywords)[:10]).encode()).hexdigest()[:8]
        
        # Complexity heuristic
        complexity_score = (
            length * 0.1 + 
            sum([has_math, has_logic, has_comparison]) * 0.3 +
            len(re.findall(r"[^\w\s]", question)) * 0.05
        )
        
        return ProblemFeatures(
            length=length,
            has_math=has_math,
            has_logic=has_logic,
            has_comparison=has_comparison,
            keyword_hash=keyword_hash,
            complexity_score=complexity_score
        )
    
    def _graph_to_metrics(self, graph: nx.DiGraph, final_score: float, generations: int) -> GraphMetrics:
        """Convert graph to metrics for storage."""
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        density = nx.density(graph) if nodes > 1 else 0.0
        
        weights = [d.get("weight", 0.5) for _, _, d in graph.edges(data=True)]
        avg_weight = np.mean(weights) if weights else 0.5
        
        return GraphMetrics(
            nodes=nodes,
            edges=edges,
            density=density,
            avg_weight=avg_weight,
            final_score=final_score,
            evolution_generations=generations
        )
    
    def predict_config(self, question: str) -> Dict[str, float]:
        """Predict optimal evolution config based on problem features and history."""
        features = self._extract_features(question)
        
        # Default config
        default_config = {
            "max_generations": 8.0,
            "population_size": 12.0,
            "mutation_rate": 0.3,
            "crossover_rate": 0.6
        }
        
        if not self.records or self.meta_net is None or torch is None:
            return default_config
        
        # Find similar problems
        similar_scores = []
        for record in self.records[-100:]:  # Recent 100 records
            similarity = self._compute_similarity(features, record.problem_features)
            if similarity > 0.3:  # Threshold for similarity
                similar_scores.append((similarity, record))
        
        if not similar_scores:
            return default_config
        
        # Use neural prediction if we have torch
        try:
            feature_vec = self._features_to_vector(features)
            with torch.no_grad():
                pred = self.meta_net(torch.tensor([feature_vec], dtype=torch.float32))
                pred = pred.cpu().numpy()[0]
            
            # Scale predictions to reasonable ranges
            config = {
                "max_generations": 4 + pred[0] * 12,  # 4-16
                "population_size": 6 + pred[1] * 18,  # 6-24
                "mutation_rate": 0.1 + pred[2] * 0.6,  # 0.1-0.7
                "crossover_rate": 0.3 + pred[3] * 0.6   # 0.3-0.9
            }
            return config
        except Exception:
            # Fallback to weighted average of similar configs
            weights = [score for score, _ in similar_scores]
            configs = [record.config_used for _, record in similar_scores]
            
            weighted_config = {}
            for key in default_config.keys():
                values = [config.get(key, default_config[key]) for config in configs]
                weighted_config[key] = np.average(values, weights=weights)
            
            return weighted_config
    
    def _compute_similarity(self, f1: ProblemFeatures, f2: ProblemFeatures) -> float:
        """Compute similarity between problem features."""
        similarity = 0.0
        
        # Length similarity (normalized)
        len_sim = 1.0 - abs(f1.length - f2.length) / max(f1.length + f2.length, 1)
        similarity += len_sim * 0.2
        
        # Boolean feature matches
        bool_matches = sum([
            f1.has_math == f2.has_math,
            f1.has_logic == f2.has_logic,
            f1.has_comparison == f2.has_comparison
        ])
        similarity += (bool_matches / 3.0) * 0.4
        
        # Keyword hash exact match
        if f1.keyword_hash == f2.keyword_hash:
            similarity += 0.3
        
        # Complexity similarity
        comp_sim = 1.0 - abs(f1.complexity_score - f2.complexity_score) / max(f1.complexity_score + f2.complexity_score, 1)
        similarity += comp_sim * 0.1
        
        return similarity
    
    def _features_to_vector(self, features: ProblemFeatures) -> List[float]:
        """Convert features to vector for neural network."""
        return [
            features.length / 50.0,  # Normalize
            float(features.has_math),
            float(features.has_logic),
            float(features.has_comparison),
            features.complexity_score / 10.0,  # Normalize
            hash(features.keyword_hash) % 1000 / 1000.0,  # Hash to [0,1]
            0.5,  # Padding
            0.5   # Padding
        ]
    
    def record_outcome(self, question: str, graph: nx.DiGraph, config: Dict[str, float], 
                      final_score: float, generations: int, success: bool):
        """Record the outcome of a problem-solving session."""
        features = self._extract_features(question)
        metrics = self._graph_to_metrics(graph, final_score, generations)
        
        record = MetaLearningRecord(
            problem_features=features,
            graph_metrics=metrics,
            config_used=config,
            success=success
        )
        
        self.records.append(record)
        
        # Trim memory if too large
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records:]
        
        self._save_memory()
        self._maybe_update_model()
    
    def _maybe_update_model(self):
        """Optionally update the meta-learning model (simplified version)."""
        if len(self.records) < 20 or self.meta_net is None or torch is None:
            return
        
        # Simple online update every 20 records
        if len(self.records) % 20 == 0:
            try:
                self._simple_model_update()
            except Exception:
                pass  # Fail gracefully
    
    def _simple_model_update(self):
        """Simple model update using recent successful records."""
        if torch is None:
            return
        
        # Get recent successful records
        recent_success = [r for r in self.records[-50:] if r.success and r.graph_metrics.final_score > 0.7]
        if len(recent_success) < 10:
            return
        
        # Prepare training data
        X = []
        y = []
        for record in recent_success:
            x_vec = self._features_to_vector(record.problem_features)
            y_vec = [
                (record.config_used.get("max_generations", 8) - 4) / 12,
                (record.config_used.get("population_size", 12) - 6) / 18,
                (record.config_used.get("mutation_rate", 0.3) - 0.1) / 0.6,
                (record.config_used.get("crossover_rate", 0.6) - 0.3) / 0.6
            ]
            X.append(x_vec)
            y.append(y_vec)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Simple gradient step
        optimizer = torch.optim.Adam(self.meta_net.parameters(), lr=0.01)
        
        self.meta_net.train()
        optimizer.zero_grad()
        pred = self.meta_net(X_tensor)
        loss = F.mse_loss(pred, y_tensor)
        loss.backward()
        optimizer.step()
        self.meta_net.eval()
    
    def _save_memory(self):
        """Save memory to disk."""
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            # Convert to dict for JSON serialization
            data = [asdict(record) for record in self.records[-100:]]  # Save last 100
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Fail gracefully
    
    def _load_memory(self):
        """Load memory from disk."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                
                records = []
                for item in data:
                    try:
                        record = MetaLearningRecord(
                            problem_features=ProblemFeatures(**item['problem_features']),
                            graph_metrics=GraphMetrics(**item['graph_metrics']),
                            config_used=item['config_used'],
                            success=item['success']
                        )
                        records.append(record)
                    except Exception:
                        continue  # Skip malformed records
                
                self.records = records
        except Exception:
            self.records = []  # Start fresh if loading fails
