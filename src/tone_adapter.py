from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class ToneStyle(Enum):
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    CONVERSATIONAL = "conversational"
    ENCOURAGING = "encouraging"
    FORMAL = "formal"


@dataclass
class ToneProfile:
    style: ToneStyle
    formality: float  # 0.0 = casual, 1.0 = formal
    encouragement: float  # 0.0 = neutral, 1.0 = very encouraging
    technical_depth: float  # 0.0 = simple, 1.0 = detailed


class ToneAdapter:
    """Adapts trace language and style based on context and user preferences."""
    
    def __init__(self, default_style: ToneStyle = ToneStyle.TECHNICAL):
        self.default_style = default_style
        
        # Context detection patterns
        self.context_patterns = {
            ToneStyle.EDUCATIONAL: [
                r"(student|learn|teach|homework|school|tutorial)",
                r"(explain|help.*understand|show.*how)",
                r"(class|grade|assignment)"
            ],
            ToneStyle.TECHNICAL: [
                r"(algorithm|implementation|optimize|system)",
                r"(technical|engineering|development)",
                r"(api|code|function|method)"
            ],
            ToneStyle.FORMAL: [
                r"(report|documentation|official|formal)",
                r"(business|professional|enterprise)",
                r"(analysis|evaluation|assessment)"
            ]
        }
        
        # Tone-specific vocabulary and phrases
        self.tone_vocabulary = {
            ToneStyle.TECHNICAL: {
                "step": "step",
                "process": "execute",
                "result": "output",
                "check": "verify",
                "good": "valid",
                "bad": "invalid",
                "start": "initialize",
                "end": "terminate"
            },
            ToneStyle.EDUCATIONAL: {
                "step": "step",
                "process": "work through",
                "result": "answer",
                "check": "double-check",
                "good": "correct",
                "bad": "incorrect",
                "start": "begin",
                "end": "finish"
            },
            ToneStyle.CONVERSATIONAL: {
                "step": "move",
                "process": "tackle",
                "result": "result",
                "check": "make sure",
                "good": "great",
                "bad": "not quite right",
                "start": "kick off",
                "end": "wrap up"
            },
            ToneStyle.ENCOURAGING: {
                "step": "step",
                "process": "work through",
                "result": "solution",
                "check": "verify",
                "good": "excellent",
                "bad": "needs adjustment",
                "start": "begin",
                "end": "complete"
            },
            ToneStyle.FORMAL: {
                "step": "phase",
                "process": "execute",
                "result": "outcome",
                "check": "validate",
                "good": "satisfactory",
                "bad": "unsatisfactory",
                "start": "commence",
                "end": "conclude"
            }
        }
        
        # Encouraging phrases by context
        self.encouragement_phrases = {
            "start": [
                "Let's break this down step by step!",
                "Great! Let's solve this together.",
                "Here we go - this looks interesting!",
                "Perfect! Let's tackle this problem."
            ],
            "progress": [
                "Nice work so far!",
                "We're making good progress!",
                "Excellent! We're on the right track.",
                "Great job! Let's continue."
            ],
            "success": [
                "Fantastic! We got it right!",
                "Excellent work! That's the correct answer.",
                "Perfect! We solved it successfully.",
                "Great job! Mission accomplished!"
            ],
            "retry": [
                "No worries, let's try a different approach!",
                "That's okay! Let's explore another path.",
                "Not to worry - we'll figure this out!",
                "Let's adjust our strategy and try again."
            ]
        }
    
    def detect_context(self, question: str, metadata: Optional[Dict] = None) -> ToneProfile:
        """Detect appropriate tone based on question context."""
        question_lower = question.lower()
        
        # Check for explicit context hints in metadata
        if metadata:
            explicit_style = metadata.get("tone_style")
            if explicit_style and isinstance(explicit_style, str):
                try:
                    style = ToneStyle(explicit_style.lower())
                    return self._create_profile(style)
                except ValueError:
                    pass
        
        # Pattern-based detection
        style_scores = {}
        for style, patterns in self.context_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    score += 1
            style_scores[style] = score
        
        # Find best matching style
        if style_scores:
            best_style = max(style_scores.keys(), key=lambda s: style_scores[s])
            if style_scores[best_style] > 0:
                return self._create_profile(best_style)
        
        # Default fallback
        return self._create_profile(self.default_style)
    
    def _create_profile(self, style: ToneStyle) -> ToneProfile:
        """Create tone profile for given style."""
        profiles = {
            ToneStyle.TECHNICAL: ToneProfile(
                style=style, formality=0.8, encouragement=0.2, technical_depth=0.9
            ),
            ToneStyle.EDUCATIONAL: ToneProfile(
                style=style, formality=0.5, encouragement=0.8, technical_depth=0.6
            ),
            ToneStyle.CONVERSATIONAL: ToneProfile(
                style=style, formality=0.3, encouragement=0.6, technical_depth=0.4
            ),
            ToneStyle.ENCOURAGING: ToneProfile(
                style=style, formality=0.4, encouragement=1.0, technical_depth=0.5
            ),
            ToneStyle.FORMAL: ToneProfile(
                style=style, formality=1.0, encouragement=0.1, technical_depth=0.8
            )
        }
        return profiles.get(style, profiles[ToneStyle.TECHNICAL])
    
    def adapt_trace_line(self, line: str, profile: ToneProfile, context: str = "progress") -> str:
        """Adapt a single trace line according to tone profile."""
        adapted_line = line
        
        # Replace vocabulary based on style
        vocab = self.tone_vocabulary.get(profile.style, {})
        for generic, specific in vocab.items():
            # Case-insensitive replacement while preserving case
            pattern = re.compile(re.escape(generic), re.IGNORECASE)
            adapted_line = pattern.sub(lambda m: self._preserve_case(m.group(), specific), adapted_line)
        
        # Add encouragement if appropriate
        if profile.encouragement > 0.5 and context in self.encouragement_phrases:
            # Occasionally add encouraging phrases (probabilistic)
            import random
            if random.random() < profile.encouragement * 0.3:  # 30% max chance
                phrase = random.choice(self.encouragement_phrases[context])
                adapted_line = f"{phrase} {adapted_line}"
        
        return adapted_line
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserve the case pattern of the original word."""
        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        else:
            return replacement.lower()
    
    def format_full_trace(self, trace: str, question: str, 
                         metadata: Optional[Dict] = None,
                         include_header: bool = True) -> str:
        """Format complete trace with appropriate tone."""
        profile = self.detect_context(question, metadata)
        lines = trace.split('\n')
        adapted_lines = []
        
        # Add contextual header
        if include_header:
            if profile.style == ToneStyle.EDUCATIONAL:
                header = "📚 Let's solve this step by step:"
            elif profile.style == ToneStyle.ENCOURAGING:
                header = "🎯 Great question! Here's how we'll tackle it:"
            elif profile.style == ToneStyle.CONVERSATIONAL:
                header = "💭 Alright, let's think through this:"
            elif profile.style == ToneStyle.FORMAL:
                header = "📊 Analysis and Solution Process:"
            else:  # Technical
                header = "🔧 DNGE Reasoning Trace:"
            
            adapted_lines.append(header)
        
        # Process each line
        context = "start"
        for i, line in enumerate(lines):
            if not line.strip():
                adapted_lines.append(line)
                continue
            
            # Determine context for this line
            if i == 0 or "Step 1" in line or "input" in line.lower():
                context = "start"
            elif "error" in line.lower() or "none" in line.lower():
                context = "retry"
            elif "verify" in line.lower() or "final" in line.lower():
                context = "success"
            else:
                context = "progress"
            
            adapted_line = self.adapt_trace_line(line, profile, context)
            adapted_lines.append(adapted_line)
        
        # Add contextual footer
        if profile.encouragement > 0.7:
            footer = "\n✨ Hope this helps! Feel free to ask if you need clarification on any step."
            adapted_lines.append(footer)
        elif profile.style == ToneStyle.FORMAL:
            footer = "\n📋 Analysis complete. All reasoning steps have been documented above."
            adapted_lines.append(footer)
        
        return '\n'.join(adapted_lines)
    
    def get_success_message(self, profile: ToneProfile, result: str) -> str:
        """Generate contextual success message."""
        if profile.style == ToneStyle.ENCOURAGING:
            return f"🎉 Excellent! The answer is: {result}"
        elif profile.style == ToneStyle.EDUCATIONAL:
            return f"✅ Great work! We found that the answer is: {result}"
        elif profile.style == ToneStyle.CONVERSATIONAL:
            return f"👍 Nice! So the answer turns out to be: {result}"
        elif profile.style == ToneStyle.FORMAL:
            return f"📊 Analysis concludes with result: {result}"
        else:  # Technical
            return f"✓ Final result: {result}"
    
    def get_failure_message(self, profile: ToneProfile) -> str:
        """Generate contextual failure message."""
        if profile.style == ToneStyle.ENCOURAGING:
            return "🤔 Hmm, we didn't get a clear answer this time, but that's okay! Let's try a different approach."
        elif profile.style == ToneStyle.EDUCATIONAL:
            return "📝 We weren't able to solve this completely. Let's review our approach and try again."
        elif profile.style == ToneStyle.CONVERSATIONAL:
            return "🤷 Didn't quite get there this time, but we gave it a good shot!"
        elif profile.style == ToneStyle.FORMAL:
            return "⚠️ Analysis did not yield a definitive result. Further investigation may be required."
        else:  # Technical
            return "❌ Execution did not produce a valid result."
