from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModel = None
    TRANSFORMERS_AVAILABLE = False


@dataclass
class Decomposition:
    graph: nx.DiGraph
    intent: str
    tokens: List[str]


# Output dim increased to 6: 5 tool hints + 1 raw confidence score
class TinyDecompMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class Decomposer:
    # Added 'sequence_solver' for the low-performing Sequences topic
    TOOL_RELIABILITY: Dict[str, float] = {
        "math_solve": 0.95,
        "arithmetic": 0.90,
        "logic_branch": 0.75,
        "compare": 0.85,
        "sequence_solver": 0.80, # New specialized tool
        "fallback": 0.20,
    }
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.embed_model = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model.eval()
            except Exception:
                self.tokenizer = None
                self.embed_model = None

        self.mlp = None
        if TORCH_AVAILABLE:
            self.mlp = TinyDecompMLP().to(self.device)
            self.mlp.eval()
            
        self.math_patterns = re.compile(r"(solve|equation|sum|difference|product|ratio|percent|\d+)", re.I)
        self.logic_patterns = re.compile(r"(if|then|implies|all|some|none|true|false|not|and|or)", re.I)
        self.compare_patterns = re.compile(r"(greater|less|more|fewer|compare|which|max|min)", re.I)
        # New pattern for sequences
        self.sequence_patterns = re.compile(r"(sequence|next number|pattern|series|nth term)", re.I)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9_]+", text)

    def _embed(self, text: str) -> Optional[List[float]]:
        if self.tokenizer is None or self.embed_model is None:
            return None
        with torch.no_grad():
            toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            outputs = self.embed_model(**toks)
            vec = outputs.last_hidden_state.mean(dim=1)
            vec = vec.squeeze(0)
            
            if vec.shape[-1] >= 128:
                return vec[:128].cpu().tolist()
            out = torch.zeros(128)
            out[: vec.shape[-1]] = vec.cpu()
            return out.tolist()

    def build_initial_graph(self, question: str) -> Decomposition:
        g = nx.DiGraph()
        tokens = self._tokenize(question)

        g.add_node("input", kind="input", text=question)

        has_math = bool(self.math_patterns.search(question))
        has_logic = bool(self.logic_patterns.search(question))
        has_compare = bool(self.compare_patterns.search(question))
        has_arith = bool(re.search(r"(add|subtract|multiply|divide|plus|minus|times|over)", question, re.I))
        has_sequence = bool(self.sequence_patterns.search(question)) # New pattern check

        intent = "general"
        if has_math:
            intent = "math"
        if has_logic and not has_math:
            intent = "logic"

        hint = [0.0] * 128
        emb = self._embed(question)
        if emb is not None:
            hint = emb

        # Initial scores for the 5 tool types: [math, logic, compare, arith, sequence]
        heuristic_scores = [
            1.0 if has_math else 0.0, 
            1.0 if has_logic else 0.0, 
            1.0 if has_compare else 0.0, 
            1.0 if has_arith else 0.0,
            1.0 if has_sequence else 0.0
        ]
        
        raw_confidence = 0.5 
        
        if self.mlp is not None and torch is not None:
            with torch.no_grad():
                x = torch.tensor([hint], dtype=torch.float32)
                mlp_out = self.mlp(x).cpu().numpy().tolist()[0]
            
            # MLP output is now size 6: 5 tool hints + 1 confidence
            mlp_tool_hints = mlp_out[:5]
            raw_confidence = mlp_out[5]
            
            # Fused scores for tool selection (0.7 Heuristic + 0.3 Neural)
            fused_tool_scores = [
                0.7 * s + 0.3 * m 
                for s, m in zip(heuristic_scores, mlp_tool_hints)
            ]
        else:
            fused_tool_scores = heuristic_scores

        tool_nodes: List[Tuple[str, float]] = []
        
        # Index map: [math, logic, compare, arith, sequence]
        # Prioritize math_solve (index 0) over arithmetic (index 3)
        if fused_tool_scores[0] > 0.3:
            tool_nodes.append(("math_solve", fused_tool_scores[0]))
        if fused_tool_scores[3] > 0.3 and not any(n == "math_solve" for n, _ in tool_nodes):
            tool_nodes.append(("arithmetic", fused_tool_scores[3]))
        if fused_tool_scores[1] > 0.3:
            tool_nodes.append(("logic_branch", fused_tool_scores[1]))
        if fused_tool_scores[2] > 0.3:
            tool_nodes.append(("compare", fused_tool_scores[2]))
        if fused_tool_scores[4] > 0.3:
            tool_nodes.append(("sequence_solver", fused_tool_scores[4])) # Add sequence solver

        for name, w in tool_nodes:
            g.add_node(name, kind="tool")
            g.add_edge("input", name, weight=float(w))

        g.add_node("aggregate", kind="aggregate")
        
        # Pass Raw Confidence to the verify node
        g.add_node("verify", kind="verify", raw_confidence=float(raw_confidence)) 
        
        for name, _ in tool_nodes:
            # Use Tool Reliability as the weight for the aggregate edge
            reliability = self.TOOL_RELIABILITY.get(name, 0.5)
            g.add_edge(name, "aggregate", weight=reliability)
            
        g.add_edge("aggregate", "verify", weight=1.0)

        if not tool_nodes:
            fallback_name = "fallback"
            fallback_reliability = self.TOOL_RELIABILITY.get(fallback_name, 0.2)
            g.add_node(fallback_name, kind="tool")
            g.add_edge("input", fallback_name, weight=fallback_reliability)
            g.add_edge("fallback", "aggregate", weight=fallback_reliability)

        return Decomposition(graph=g, intent=intent, tokens=tokens)