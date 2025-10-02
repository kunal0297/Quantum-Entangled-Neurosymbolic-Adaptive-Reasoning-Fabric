from __future__ import annotations

from typing import Any, Dict, List
import networkx as nx


def format_trace(g: nx.DiGraph, ctx: Dict[str, Any], evolution_log: List[dict] | None = None) -> str:
    lines: List[str] = []
    lines.append("DNGE Reasoning Trace:")
    for node in nx.topological_sort(g):
        data = ctx.get(node, {})
        lines.append(f"- Step {node}: kind={g.nodes[node].get('kind')} -> {str(data.get('result'))}")
    if evolution_log:
        lines.append("Evolution:")
        for item in evolution_log[-5:]:
            lines.append(f"  gen={item.get('generation')} best={round(item.get('best_score',0.0),4)}")
    return "\n".join(lines)


