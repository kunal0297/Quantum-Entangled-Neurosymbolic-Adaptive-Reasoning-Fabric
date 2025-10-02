from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

import networkx as nx


class FeedbackType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    SUGGEST_TOOL = "suggest_tool"
    SUGGEST_PATH = "suggest_path"
    INCREASE_WEIGHT = "increase_weight"
    DECREASE_WEIGHT = "decrease_weight"


@dataclass
class FeedbackItem:
    type: FeedbackType
    target: str  # Node name or edge or general
    message: str
    confidence: float = 1.0


class FeedbackHandler:
    """Handles interactive human feedback during reasoning process."""
    
    def __init__(self, interactive: bool = False):
        self.interactive = interactive
        self.feedback_history: List[FeedbackItem] = []
        
        # Feedback patterns for parsing text input
        self.patterns = {
            FeedbackType.APPROVE: [
                r"(good|correct|right|approve|yes|ok)",
                r"this.*looks.*good",
                r"continue"
            ],
            FeedbackType.REJECT: [
                r"(wrong|incorrect|bad|reject|no)",
                r"this.*wrong",
                r"try.*different"
            ],
            FeedbackType.SUGGEST_TOOL: [
                r"use.*(\w+)",
                r"try.*(\w+)",
                r"should.*use.*(\w+)"
            ],
            FeedbackType.SUGGEST_PATH: [
                r"skip.*(\w+)",
                r"go.*directly.*(\w+)",
                r"bypass.*(\w+)"
            ],
            FeedbackType.INCREASE_WEIGHT: [
                r"emphasize.*(\w+)",
                r"focus.*(\w+)",
                r"increase.*(\w+)",
                r"more.*(\w+)"
            ],
            FeedbackType.DECREASE_WEIGHT: [
                r"reduce.*(\w+)",
                r"less.*(\w+)",
                r"decrease.*(\w+)",
                r"ignore.*(\w+)"
            ]
        }
    
    def parse_feedback(self, text: str) -> List[FeedbackItem]:
        """Parse natural language feedback into structured feedback items."""
        text = text.lower().strip()
        if not text:
            return []
        
        feedback_items = []
        
        for feedback_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Extract target if present in groups
                    target = match.group(1) if match.groups() else "general"
                    
                    # Determine confidence based on pattern strength
                    confidence = 0.9 if len(match.groups()) > 0 else 0.7
                    
                    feedback_items.append(FeedbackItem(
                        type=feedback_type,
                        target=target,
                        message=text,
                        confidence=confidence
                    ))
                    break  # Only one match per type
        
        # If no patterns matched, treat as general feedback
        if not feedback_items:
            # Guess intent based on sentiment
            if any(word in text for word in ["good", "correct", "right", "yes"]):
                feedback_items.append(FeedbackItem(
                    type=FeedbackType.APPROVE,
                    target="general",
                    message=text,
                    confidence=0.5
                ))
            elif any(word in text for word in ["wrong", "incorrect", "bad", "no"]):
                feedback_items.append(FeedbackItem(
                    type=FeedbackType.REJECT,
                    target="general",
                    message=text,
                    confidence=0.5
                ))
        
        return feedback_items
    
    def apply_feedback_to_graph(self, graph: nx.DiGraph, feedback_items: List[FeedbackItem]) -> nx.DiGraph:
        """Apply feedback to modify graph structure and weights."""
        modified_graph = graph.copy()
        
        for item in feedback_items:
            if item.confidence < 0.3:  # Skip low-confidence feedback
                continue
            
            if item.type == FeedbackType.REJECT:
                self._handle_rejection(modified_graph, item)
            elif item.type == FeedbackType.SUGGEST_TOOL:
                self._handle_tool_suggestion(modified_graph, item)
            elif item.type == FeedbackType.SUGGEST_PATH:
                self._handle_path_suggestion(modified_graph, item)
            elif item.type == FeedbackType.INCREASE_WEIGHT:
                self._handle_weight_increase(modified_graph, item)
            elif item.type == FeedbackType.DECREASE_WEIGHT:
                self._handle_weight_decrease(modified_graph, item)
            # APPROVE doesn't need explicit handling - it's positive reinforcement
        
        return modified_graph
    
    def _handle_rejection(self, graph: nx.DiGraph, item: FeedbackItem):
        """Handle rejection feedback by reducing weights or removing edges."""
        if item.target == "general":
            # General rejection - reduce all edge weights slightly
            for u, v, d in graph.edges(data=True):
                d["weight"] = max(0.1, d.get("weight", 0.5) * 0.8)
        else:
            # Specific node rejection
            if graph.has_node(item.target):
                # Reduce incoming edge weights
                for pred in graph.predecessors(item.target):
                    if graph.has_edge(pred, item.target):
                        graph[pred][item.target]["weight"] = max(0.1, 
                            graph[pred][item.target].get("weight", 0.5) * 0.5)
    
    def _handle_tool_suggestion(self, graph: nx.DiGraph, item: FeedbackItem):
        """Handle tool suggestion by adding or strengthening tool nodes."""
        tool_name = item.target
        
        # Add tool node if it doesn't exist
        if not graph.has_node(tool_name):
            graph.add_node(tool_name, kind="tool", source="feedback")
        
        # Connect to input if not already connected
        if graph.has_node("input") and not graph.has_edge("input", tool_name):
            graph.add_edge("input", tool_name, weight=0.8, source="feedback")
        
        # Connect to aggregate if it exists
        if graph.has_node("aggregate") and not graph.has_edge(tool_name, "aggregate"):
            graph.add_edge(tool_name, "aggregate", weight=0.7, source="feedback")
    
    def _handle_path_suggestion(self, graph: nx.DiGraph, item: FeedbackItem):
        """Handle path suggestion by adding direct connections."""
        target_node = item.target
        
        if graph.has_node(target_node):
            # Add skip connections to suggested node
            if graph.has_node("input") and not graph.has_edge("input", target_node):
                graph.add_edge("input", target_node, weight=0.6, source="feedback_skip")
    
    def _handle_weight_increase(self, graph: nx.DiGraph, item: FeedbackItem):
        """Increase weights related to specified node."""
        target_node = item.target
        
        if graph.has_node(target_node):
            # Increase incoming edge weights
            for pred in graph.predecessors(target_node):
                if graph.has_edge(pred, target_node):
                    current_weight = graph[pred][target_node].get("weight", 0.5)
                    graph[pred][target_node]["weight"] = min(1.0, current_weight * 1.3)
    
    def _handle_weight_decrease(self, graph: nx.DiGraph, item: FeedbackItem):
        """Decrease weights related to specified node."""
        target_node = item.target
        
        if graph.has_node(target_node):
            # Decrease incoming edge weights
            for pred in graph.predecessors(target_node):
                if graph.has_edge(pred, target_node):
                    current_weight = graph[pred][target_node].get("weight", 0.5)
                    graph[pred][target_node]["weight"] = max(0.1, current_weight * 0.7)
    
    def create_feedback_fitness_modifier(self, feedback_items: List[FeedbackItem]) -> Callable[[float], float]:
        """Create a fitness modifier based on accumulated feedback."""
        def fitness_modifier(base_fitness: float) -> float:
            modifier = 1.0
            
            # Count positive vs negative feedback
            positive_count = sum(1 for item in feedback_items 
                               if item.type == FeedbackType.APPROVE and item.confidence > 0.5)
            negative_count = sum(1 for item in feedback_items 
                               if item.type == FeedbackType.REJECT and item.confidence > 0.5)
            
            # Adjust based on feedback sentiment
            if positive_count > negative_count:
                modifier = 1.0 + 0.1 * (positive_count - negative_count)
            elif negative_count > positive_count:
                modifier = 1.0 - 0.1 * (negative_count - positive_count)
            
            return base_fitness * max(0.1, modifier)
        
        return fitness_modifier
    
    def get_interactive_feedback(self, trace: str, graph: nx.DiGraph) -> List[FeedbackItem]:
        """Get interactive feedback from user (if interactive mode enabled)."""
        if not self.interactive:
            return []
        
        print("\n" + "="*50)
        print("INTERMEDIATE REASONING TRACE:")
        print(trace)
        print("\nGraph nodes:", list(graph.nodes()))
        print("Graph edges:", list(graph.edges()))
        print("\nFeedback options:")
        print("- 'good' or 'approve' - current reasoning is correct")
        print("- 'wrong' or 'reject' - current reasoning is incorrect")
        print("- 'use [tool]' - suggest using a specific tool")
        print("- 'emphasize [node]' - increase focus on a node")
        print("- 'reduce [node]' - decrease focus on a node")
        print("- Enter to continue without feedback")
        print("="*50)
        
        try:
            user_input = input("Your feedback: ").strip()
            if user_input:
                feedback_items = self.parse_feedback(user_input)
                self.feedback_history.extend(feedback_items)
                return feedback_items
        except (KeyboardInterrupt, EOFError):
            print("\nFeedback interrupted, continuing...")
        
        return []
    
    def get_feedback_summary(self) -> Dict[str, int]:
        """Get summary of feedback history."""
        summary = {}
        for item in self.feedback_history:
            key = f"{item.type.value}_{item.target}"
            summary[key] = summary.get(key, 0) + 1
        return summary
