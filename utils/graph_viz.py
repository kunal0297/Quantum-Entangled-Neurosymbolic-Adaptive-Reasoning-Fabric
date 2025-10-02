from __future__ import annotations

import io
import os
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def render_graph_png(g: nx.DiGraph, output_path: Optional[str] = None, 
                    context: Optional[Dict[str, Any]] = None,
                    uncertainties: Optional[Dict[str, float]] = None) -> bytes:
    """Render graph as PNG with enhanced visualization features."""
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(g, seed=42, k=2, iterations=50)
    
    # Color nodes by type and success
    node_colors = []
    node_sizes = []
    
    for node in g.nodes():
        node_data = g.nodes[node]
        node_kind = node_data.get("kind", "unknown")
        
        # Base color by type
        if node_kind == "input":
            base_color = "#4CAF50"  # Green
        elif node_kind == "tool":
            base_color = "#2196F3"  # Blue
        elif node_kind == "aggregate":
            base_color = "#FF9800"  # Orange
        elif node_kind == "verify":
            base_color = "#9C27B0"  # Purple
        else:
            base_color = "#757575"  # Gray
        
        # Modify color based on execution success
        if context and node in context:
            result = context[node].get("result")
            if result is not None:
                # Success - brighter color
                base_color = base_color.replace("#", "#FF") if len(base_color) == 7 else base_color
            else:
                # Failed - darker/redder
                base_color = "#F44336"  # Red
        
        # Modify color based on uncertainty
        if uncertainties and node in uncertainties:
            uncertainty = uncertainties[node]
            if uncertainty > 0.7:
                # High uncertainty - add yellow tint
                base_color = "#FFC107"  # Amber
        
        node_colors.append(base_color)
        
        # Size based on importance (in-degree + out-degree)
        importance = g.in_degree(node) + g.out_degree(node)
        node_sizes.append(max(300, 100 * importance))
    
    # Draw nodes
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw node labels
    labels = {}
    for node in g.nodes():
        # Truncate long labels
        label = node[:8] + "..." if len(node) > 8 else node
        if context and node in context:
            result = context[node].get("result")
            if result is not None:
                # Add checkmark for successful nodes
                label += " ✓"
            else:
                label += " ✗"
        labels[node] = label
    
    nx.draw_networkx_labels(g, pos, labels, font_size=8, font_weight='bold')
    
    # Color edges by weight and source
    edge_colors = []
    edge_styles = []
    edge_widths = []
    
    for u, v, data in g.edges(data=True):
        weight = data.get("weight", 0.5)
        source = data.get("source", "original")
        
        # Color by weight
        if weight > 0.7:
            color = "#4CAF50"  # Green for high confidence
        elif weight > 0.4:
            color = "#2196F3"  # Blue for medium confidence
        else:
            color = "#FF9800"  # Orange for low confidence
        
        # Style by source
        if source == "analogy":
            style = "dashed"
        elif "feedback" in source:
            style = "dotted"
        elif "uncertainty" in source:
            style = "dashdot"
        else:
            style = "solid"
        
        edge_colors.append(color)
        edge_styles.append(style)
        edge_widths.append(max(0.5, weight * 3))
    
    # Draw edges with different styles
    unique_styles = list(set(edge_styles))
    for style in unique_styles:
        edge_list = [(u, v) for u, v, d in g.edges(data=True) 
                    if (d.get("source", "original") == "analogy" and style == "dashed") or
                       ("feedback" in d.get("source", "") and style == "dotted") or
                       ("uncertainty" in d.get("source", "") and style == "dashdot") or
                       (d.get("source", "original") not in ["analogy", "feedback", "uncertainty"] and style == "solid")]
        
        if edge_list:
            edge_colors_subset = [edge_colors[i] for i, (u, v) in enumerate(g.edges()) if (u, v) in edge_list]
            edge_widths_subset = [edge_widths[i] for i, (u, v) in enumerate(g.edges()) if (u, v) in edge_list]
            
            nx.draw_networkx_edges(g, pos, edgelist=edge_list, 
                                 edge_color=edge_colors_subset,
                                 width=edge_widths_subset,
                                 style=style, arrows=True, arrowsize=20,
                                 alpha=0.7)
    
    # Add edge labels for weights
    edge_labels = {}
    for u, v, data in g.edges(data=True):
        weight = data.get("weight", 0.5)
        if weight != 0.5:  # Only show non-default weights
            edge_labels[(u, v)] = f"{weight:.2f}"
    
    if edge_labels:
        nx.draw_networkx_edge_labels(g, pos, edge_labels, font_size=6)
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color="#4CAF50", label="Input Node"),
        mpatches.Patch(color="#2196F3", label="Tool Node"),
        mpatches.Patch(color="#FF9800", label="Aggregate Node"),
        mpatches.Patch(color="#9C27B0", label="Verify Node"),
        mpatches.Patch(color="#F44336", label="Failed Execution"),
        mpatches.Patch(color="#FFC107", label="High Uncertainty")
    ]
    
    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0, 1))
    
    # Title and formatting
    plt.title("DNGE Reasoning Graph", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    
    # Save or return bytes
    if output_path:
        plt.savefig(output_path, format="png", dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to: {output_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def render_evolution_progress(evolution_log: list, output_path: Optional[str] = None) -> bytes:
    """Render evolution progress as a line chart."""
    if not evolution_log:
        # Create empty plot
        plt.figure(figsize=(8, 5))
        plt.text(0.5, 0.5, "No evolution data available", 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title("Evolution Progress")
    else:
        generations = [item.get("generation", i) for i, item in enumerate(evolution_log)]
        scores = [item.get("best_score", 0.0) for item in evolution_log]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, scores, 'b-o', linewidth=2, markersize=6)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness Score")
        plt.title("DNGE Evolution Progress")
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(generations) > 1:
            z = np.polyfit(generations, scores, 1)
            p = np.poly1d(z)
            plt.plot(generations, p(generations), "r--", alpha=0.8, label="Trend")
            plt.legend()
    
    if output_path:
        plt.savefig(output_path, format="png", dpi=150, bbox_inches='tight')
        print(f"Evolution progress saved to: {output_path}")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf.read()


def create_interactive_html(graph: nx.DiGraph, context: Dict[str, Any], 
                          trace: str, evolution_log: list,
                          output_path: str = "reasoning_report.html"):
    """Create an interactive HTML report with graph visualization."""
    
    # Generate base64 encoded images
    import base64
    
    graph_png = render_graph_png(graph, context=context)
    graph_b64 = base64.b64encode(graph_png).decode()
    
    evolution_png = render_evolution_progress(evolution_log)
    evolution_b64 = base64.b64encode(evolution_png).decode()
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>DNGE Reasoning Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .graph-container {{ text-align: center; margin: 20px 0; }}
            .trace {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; font-family: monospace; white-space: pre-wrap; }}
            .stats {{ display: flex; justify-content: space-around; background-color: #e3f2fd; padding: 15px; border-radius: 5px; }}
            .stat {{ text-align: center; }}
            .stat h3 {{ margin: 0; color: #1976d2; }}
            .collapsible {{ background-color: #2196f3; color: white; cursor: pointer; padding: 10px; border: none; text-align: left; outline: none; font-size: 15px; width: 100%; }}
            .collapsible:hover {{ background-color: #1976d2; }}
            .content {{ padding: 0 18px; display: none; overflow: hidden; background-color: #f1f1f1; }}
        </style>
        <script>
            function toggleContent(element) {{
                var content = element.nextElementSibling;
                if (content.style.display === "block") {{
                    content.style.display = "none";
                }} else {{
                    content.style.display = "block";
                }}
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 NEURAGRAPH / DNGE Reasoning Report</h1>
                <p><em>Dynamic Neurosymbolic Graph Evolution in Action</em></p>
            </div>
            
            <div class="section">
                <h2>📊 Performance Statistics</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>{len(graph.nodes())}</h3>
                        <p>Graph Nodes</p>
                    </div>
                    <div class="stat">
                        <h3>{len(graph.edges())}</h3>
                        <p>Graph Edges</p>
                    </div>
                    <div class="stat">
                        <h3>{len(evolution_log)}</h3>
                        <p>Evolution Generations</p>
                    </div>
                    <div class="stat">
                        <h3>{evolution_log[-1].get('best_score', 0.0):.3f}</h3>
                        <p>Final Fitness</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 Evolved Reasoning Graph</h2>
                <div class="graph-container">
                    <img src="data:image/png;base64,{graph_b64}" alt="Reasoning Graph" style="max-width: 100%; height: auto;"/>
                </div>
                <p><strong>Legend:</strong> Green=Input, Blue=Tools, Orange=Aggregate, Purple=Verify. 
                Edge thickness indicates confidence, style indicates source (dashed=analogy, dotted=feedback, dash-dot=uncertainty).</p>
            </div>
            
            <div class="section">
                <h2>📈 Evolution Progress</h2>
                <div class="graph-container">
                    <img src="data:image/png;base64,{evolution_b64}" alt="Evolution Progress" style="max-width: 100%; height: auto;"/>
                </div>
            </div>
            
            <div class="section">
                <h2>📝 Reasoning Trace</h2>
                <button class="collapsible" onclick="toggleContent(this)">Click to expand full reasoning trace</button>
                <div class="content">
                    <div class="trace">{trace}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 DNGE Innovation Highlights</h2>
                <ul>
                    <li><strong>Meta-Learning:</strong> Graph evolution optimized based on problem history</li>
                    <li><strong>Analogical Reasoning:</strong> Reuses successful patterns from similar problems</li>
                    <li><strong>Uncertainty Modeling:</strong> Creative exploration of alternative reasoning paths</li>
                    <li><strong>Feedback Integration:</strong> Human guidance incorporated into graph evolution</li>
                    <li><strong>Neurosymbolic Fusion:</strong> Neural hints + symbolic execution + evolutionary optimization</li>
                </ul>
            </div>
            
            <div class="section">
                <p style="text-align: center; color: #666; font-size: 0.9em;">
                    Generated by NEURAGRAPH (DNGE) - Dynamic Neurosymbolic Graph Evolution<br>
                    A revolutionary approach to explainable AI reasoning
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive reasoning report saved to: {output_path}")
    return output_path


