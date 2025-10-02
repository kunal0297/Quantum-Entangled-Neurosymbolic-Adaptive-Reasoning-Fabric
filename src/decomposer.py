from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import networkx as nx

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:
    torch = None
    nn = None
    F = None

try:
    from transformers import AutoTokenizer, AutoModel
except Exception:
    AutoTokenizer = None
    AutoModel = None


@dataclass
class Decomposition:
    graph: nx.DiGraph
    intent: str
    tokens: List[str]


class TinyDecompMLP(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return torch.sigmoid(self.fc2(x))


class Decomposer:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.tokenizer = None
        self.embed_model = None
        if AutoTokenizer and AutoModel:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
                self.embed_model.eval()
            except Exception:
                self.tokenizer = None
                self.embed_model = None

        self.mlp = None
        if torch is not None and nn is not None:
            self.mlp = TinyDecompMLP().to(self.device)
            self.mlp.eval()
        self.math_patterns = re.compile(r"(solve|equation|sum|difference|product|ratio|percent|\d+)", re.I)
        self.logic_patterns = re.compile(r"(if|then|implies|all|some|none|true|false|not|and|or)", re.I)
        self.compare_patterns = re.compile(r"(greater|less|more|fewer|compare|which|max|min)", re.I)

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

        # Node: input
        g.add_node("input", kind="input", text=question)

        has_math = bool(self.math_patterns.search(question))
        has_logic = bool(self.logic_patterns.search(question))
        has_compare = bool(self.compare_patterns.search(question))
        has_arith = bool(re.search(r"(add|subtract|multiply|divide|plus|minus|times|over)", question, re.I))

        intent = "general"
        if has_math:
            intent = "math"
        if has_logic and not has_math:
            intent = "logic"

        hint = [0.0] * 128
        emb = self._embed(question)
        if emb is not None:
            hint = emb

        scores = [1.0 if has_math else 0.0, 1.0 if has_logic else 0.0, 1.0 if has_compare else 0.0, 1.0 if has_arith else 0.0]
        if self.mlp is not None and torch is not None:
            with torch.no_grad():
                x = torch.tensor([hint], dtype=torch.float32)
                mlp_out = self.mlp(x).cpu().numpy().tolist()[0]
                scores = [0.7 * s + 0.3 * m for s, m in zip(scores, mlp_out)]

        tool_nodes: List[Tuple[str, float]] = []
        if scores[0] > 0.3:
            tool_nodes.append(("math_solve", scores[0]))
        if scores[3] > 0.3 and not any(n == "math_solve" for n, _ in tool_nodes):
            tool_nodes.append(("arithmetic", scores[3]))
        if scores[1] > 0.3:
            tool_nodes.append(("logic_branch", scores[1]))
        if scores[2] > 0.3:
            tool_nodes.append(("compare", scores[2]))

        for name, w in tool_nodes:
            g.add_node(name, kind="tool")
            g.add_edge("input", name, weight=float(w))

        g.add_node("aggregate", kind="aggregate")
        g.add_node("verify", kind="verify")
        for name, _ in tool_nodes:
            g.add_edge(name, "aggregate", weight=0.5)
        g.add_edge("aggregate", "verify", weight=1.0)

        if not tool_nodes:
            g.add_node("fallback", kind="tool")
            g.add_edge("input", "fallback", weight=0.2)
            g.add_edge("fallback", "aggregate", weight=0.2)

        return Decomposition(graph=g, intent=intent, tokens=tokens)


