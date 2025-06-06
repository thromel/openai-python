"""Specialized processors for streaming validation."""

import re
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Pattern
from dataclasses import dataclass

from .stream_validator import (
    StreamingContract, StreamChunk, StreamValidationResult, 
    ViolationSeverity, ChunkValidationMode
)


class TokenProcessor(StreamingContract):
    """Processes and validates individual tokens in the stream."""
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        forbidden_tokens: Optional[Set[str]] = None,
        required_tokens: Optional[Set[str]] = None,
        token_rate_limit: Optional[float] = None,  # tokens per second
    ):
        super().__init__(
            name="TokenProcessor",
            description="Validates token-level constraints in streaming responses",
            validation_mode=ChunkValidationMode.IMMEDIATE
        )
        self.max_tokens = max_tokens
        self.forbidden_tokens = forbidden_tokens or set()
        self.required_tokens = required_tokens or set()
        self.token_rate_limit = token_rate_limit
        
        self.token_count = 0
        self.start_time = None
        self.forbidden_found: Set[str] = set()
        self.required_found: Set[str] = set()
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate tokens in the chunk."""
        if self.start_time is None:
            self.start_time = chunk.timestamp
        
        # Simple tokenization (split by whitespace)
        tokens = chunk.content.split()
        self.token_count += len(tokens)
        
        # Check forbidden tokens
        chunk_forbidden = set(tokens) & self.forbidden_tokens
        if chunk_forbidden:
            self.forbidden_found.update(chunk_forbidden)
            return StreamValidationResult(
                is_valid=False,
                should_terminate=True,
                violation_severity=ViolationSeverity.CRITICAL,
                violation_message=f"Forbidden tokens detected: {', '.join(chunk_forbidden)}"
            )
        
        # Check max tokens
        if self.max_tokens and self.token_count > self.max_tokens:
            return StreamValidationResult(
                is_valid=False,
                should_terminate=True,
                violation_severity=ViolationSeverity.HIGH,
                violation_message=f"Token limit exceeded: {self.token_count} > {self.max_tokens}"
            )
        
        # Check token rate
        if self.token_rate_limit and self.start_time:
            elapsed = chunk.timestamp - self.start_time
            if elapsed > 0:
                current_rate = self.token_count / elapsed
                if current_rate > self.token_rate_limit:
                    return StreamValidationResult(
                        is_valid=False,
                        violation_severity=ViolationSeverity.MEDIUM,
                        violation_message=f"Token rate too high: {current_rate:.1f} > {self.token_rate_limit}"
                    )
        
        # Check for required tokens
        chunk_required = set(tokens) & self.required_tokens
        self.required_found.update(chunk_required)
        
        return StreamValidationResult(is_valid=True)
    
    async def validate_incremental(
        self,
        cumulative_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate cumulative token requirements."""
        missing_required = self.required_tokens - self.required_found
        if missing_required:
            # Check if we have all required tokens in the cumulative content
            all_tokens = set(cumulative_content.split())
            still_missing = missing_required - all_tokens
            
            if still_missing and len(cumulative_content) > 1000:  # Only enforce after substantial content
                return StreamValidationResult(
                    is_valid=False,
                    violation_severity=ViolationSeverity.MEDIUM,
                    violation_message=f"Missing required tokens: {', '.join(still_missing)}"
                )
        
        return StreamValidationResult(is_valid=True)


class ContentFilter(StreamingContract):
    """Filters content based on patterns and policies."""
    
    def __init__(
        self,
        blocked_patterns: Optional[List[str]] = None,
        allowed_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
        stop_on_violation: bool = True,
    ):
        super().__init__(
            name="ContentFilter",
            description="Filters content based on patterns and policies",
            validation_mode=ChunkValidationMode.IMMEDIATE
        )
        
        flags = 0 if case_sensitive else re.IGNORECASE
        self.blocked_patterns = [re.compile(p, flags) for p in (blocked_patterns or [])]
        self.allowed_patterns = [re.compile(p, flags) for p in (allowed_patterns or [])]
        self.stop_on_violation = stop_on_violation
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Filter content in real-time."""
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            match = pattern.search(chunk.cumulative_content)
            if match:
                return StreamValidationResult(
                    is_valid=False,
                    should_terminate=self.stop_on_violation,
                    violation_severity=ViolationSeverity.HIGH,
                    violation_message=f"Blocked pattern detected: {pattern.pattern}",
                    auto_fix_suggestion="[CONTENT FILTERED]"
                )
        
        # Check allowed patterns (if specified, content must match at least one)
        if self.allowed_patterns:
            has_allowed = any(pattern.search(chunk.cumulative_content) for pattern in self.allowed_patterns)
            if not has_allowed and len(chunk.cumulative_content) > 100:  # Only enforce after some content
                return StreamValidationResult(
                    is_valid=False,
                    violation_severity=ViolationSeverity.MEDIUM,
                    violation_message="Content does not match any allowed patterns"
                )
        
        return StreamValidationResult(is_valid=True)


class PatternMatcher(StreamingContract):
    """Matches specific patterns and validates their presence/absence."""
    
    def __init__(
        self,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None,
        pattern_limits: Optional[Dict[str, int]] = None,
        case_sensitive: bool = False,
    ):
        super().__init__(
            name="PatternMatcher",
            description="Validates presence/absence of specific patterns",
            validation_mode=ChunkValidationMode.BUFFERED
        )
        
        flags = 0 if case_sensitive else re.IGNORECASE
        self.required_patterns = [re.compile(p, flags) for p in (required_patterns or [])]
        self.forbidden_patterns = [re.compile(p, flags) for p in (forbidden_patterns or [])]
        self.pattern_limits = pattern_limits or {}
        
        # Compile pattern limits
        self.limit_patterns = {
            re.compile(pattern, flags): limit 
            for pattern, limit in self.pattern_limits.items()
        }
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate patterns in real-time."""
        content = chunk.cumulative_content
        
        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            if pattern.search(content):
                return StreamValidationResult(
                    is_valid=False,
                    should_terminate=True,
                    violation_severity=ViolationSeverity.CRITICAL,
                    violation_message=f"Forbidden pattern found: {pattern.pattern}"
                )
        
        # Check pattern limits
        for pattern, limit in self.limit_patterns.items():
            matches = len(pattern.findall(content))
            if matches > limit:
                return StreamValidationResult(
                    is_valid=False,
                    violation_severity=ViolationSeverity.HIGH,
                    violation_message=f"Pattern limit exceeded: {pattern.pattern} found {matches} times (limit: {limit})"
                )
        
        return StreamValidationResult(is_valid=True)
    
    async def validate_incremental(
        self,
        cumulative_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate required patterns in complete content."""
        # Only check required patterns when we have substantial content
        if len(cumulative_content) < 50:
            return StreamValidationResult(is_valid=True)
        
        missing_patterns = []
        for pattern in self.required_patterns:
            if not pattern.search(cumulative_content):
                missing_patterns.append(pattern.pattern)
        
        if missing_patterns:
            return StreamValidationResult(
                is_valid=False,
                violation_severity=ViolationSeverity.MEDIUM,
                violation_message=f"Missing required patterns: {', '.join(missing_patterns)}"
            )
        
        return StreamValidationResult(is_valid=True)


class JSONValidator(StreamingContract):
    """Validates JSON structure as it's being streamed."""
    
    def __init__(
        self,
        require_complete_json: bool = True,
        max_nesting_depth: Optional[int] = None,
        required_fields: Optional[List[str]] = None,
        forbidden_fields: Optional[List[str]] = None,
    ):
        super().__init__(
            name="JSONValidator",
            description="Validates JSON structure in streaming responses",
            validation_mode=ChunkValidationMode.BUFFERED
        )
        self.require_complete_json = require_complete_json
        self.max_nesting_depth = max_nesting_depth
        self.required_fields = required_fields or []
        self.forbidden_fields = forbidden_fields or []
        
        self.bracket_stack = []
        self.in_string = False
        self.escape_next = False
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate JSON structure incrementally."""
        # Track bracket balance and string state
        for char in chunk.content:
            if self.escape_next:
                self.escape_next = False
                continue
            
            if char == '\\' and self.in_string:
                self.escape_next = True
                continue
            
            if char == '"' and not self.escape_next:
                self.in_string = not self.in_string
                continue
            
            if not self.in_string:
                if char in '{[':
                    self.bracket_stack.append(char)
                elif char in '}]':
                    if not self.bracket_stack:
                        return StreamValidationResult(
                            is_valid=False,
                            violation_severity=ViolationSeverity.HIGH,
                            violation_message="Unmatched closing bracket in JSON"
                        )
                    
                    expected = '{' if char == '}' else '['
                    if self.bracket_stack[-1] != expected:
                        return StreamValidationResult(
                            is_valid=False,
                            violation_severity=ViolationSeverity.HIGH,
                            violation_message="Mismatched brackets in JSON"
                        )
                    
                    self.bracket_stack.pop()
        
        # Check nesting depth
        if self.max_nesting_depth and len(self.bracket_stack) > self.max_nesting_depth:
            return StreamValidationResult(
                is_valid=False,
                violation_severity=ViolationSeverity.MEDIUM,
                violation_message=f"JSON nesting too deep: {len(self.bracket_stack)} > {self.max_nesting_depth}"
            )
        
        return StreamValidationResult(is_valid=True)
    
    async def validate_incremental(
        self,
        cumulative_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Validate complete JSON when possible."""
        content = cumulative_content.strip()
        if not content:
            return StreamValidationResult(is_valid=True)
        
        # Try to parse as JSON if it looks complete
        if content.startswith(('{', '[')) and content.endswith(('}', ']')):
            try:
                data = json.loads(content)
                
                # Check required/forbidden fields
                if isinstance(data, dict):
                    # Check forbidden fields
                    forbidden_found = set(data.keys()) & set(self.forbidden_fields)
                    if forbidden_found:
                        return StreamValidationResult(
                            is_valid=False,
                            violation_severity=ViolationSeverity.HIGH,
                            violation_message=f"Forbidden fields found: {', '.join(forbidden_found)}"
                        )
                    
                    # Check required fields
                    missing_required = set(self.required_fields) - set(data.keys())
                    if missing_required:
                        return StreamValidationResult(
                            is_valid=False,
                            violation_severity=ViolationSeverity.MEDIUM,
                            violation_message=f"Missing required fields: {', '.join(missing_required)}"
                        )
                
                return StreamValidationResult(is_valid=True)
                
            except json.JSONDecodeError as e:
                if self.require_complete_json:
                    return StreamValidationResult(
                        is_valid=False,
                        violation_severity=ViolationSeverity.HIGH,
                        violation_message=f"Invalid JSON: {str(e)}"
                    )
        
        return StreamValidationResult(is_valid=True)


class LengthMonitor(StreamingContract):
    """Monitors content length with real-time limits."""
    
    def __init__(
        self,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        warn_threshold: Optional[int] = None,
        truncate_on_limit: bool = False,
    ):
        super().__init__(
            name="LengthMonitor",
            description="Monitors content length in real-time",
            validation_mode=ChunkValidationMode.IMMEDIATE
        )
        self.max_length = max_length
        self.min_length = min_length
        self.warn_threshold = warn_threshold
        self.truncate_on_limit = truncate_on_limit
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Monitor length in real-time."""
        current_length = len(chunk.cumulative_content)
        
        # Check maximum length
        if self.max_length and current_length > self.max_length:
            auto_fix = None
            if self.truncate_on_limit:
                auto_fix = chunk.cumulative_content[:self.max_length]
            
            return StreamValidationResult(
                is_valid=False,
                should_terminate=True,
                violation_severity=ViolationSeverity.HIGH,
                violation_message=f"Content too long: {current_length} > {self.max_length}",
                auto_fix_suggestion=auto_fix
            )
        
        # Check warning threshold
        if self.warn_threshold and current_length > self.warn_threshold:
            return StreamValidationResult(
                is_valid=True,  # Still valid, just a warning
                violation_severity=ViolationSeverity.LOW,
                violation_message=f"Content approaching length limit: {current_length} > {self.warn_threshold}"
            )
        
        return StreamValidationResult(is_valid=True)
    
    async def validate_incremental(
        self,
        cumulative_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Check minimum length requirement."""
        if self.min_length and len(cumulative_content) < self.min_length:
            return StreamValidationResult(
                is_valid=False,
                violation_severity=ViolationSeverity.MEDIUM,
                violation_message=f"Content too short: {len(cumulative_content)} < {self.min_length}"
            )
        
        return StreamValidationResult(is_valid=True)


class SentimentAnalyzer(StreamingContract):
    """Analyzes sentiment in real-time (simplified implementation)."""
    
    def __init__(
        self,
        min_sentiment: float = -1.0,
        max_sentiment: float = 1.0,
        sentiment_window: int = 100,  # characters
    ):
        super().__init__(
            name="SentimentAnalyzer",
            description="Analyzes sentiment in streaming content",
            validation_mode=ChunkValidationMode.BUFFERED
        )
        self.min_sentiment = min_sentiment
        self.max_sentiment = max_sentiment
        self.sentiment_window = sentiment_window
        
        # Simple sentiment word lists
        self.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'positive', 'happy', 'joy'}
        self.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'negative', 'sad', 'angry', 'hate', 'disgusting'}
    
    def _calculate_sentiment(self, text: str) -> float:
        """Simple sentiment calculation."""
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        return (positive_count - negative_count) / total_sentiment_words
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Analyze sentiment in real-time."""
        # Only analyze if we have enough content
        if len(chunk.cumulative_content) < self.sentiment_window:
            return StreamValidationResult(is_valid=True)
        
        # Analyze recent content
        recent_content = chunk.cumulative_content[-self.sentiment_window:]
        sentiment = self._calculate_sentiment(recent_content)
        
        if sentiment < self.min_sentiment:
            return StreamValidationResult(
                is_valid=False,
                violation_severity=ViolationSeverity.MEDIUM,
                violation_message=f"Sentiment too negative: {sentiment:.2f} < {self.min_sentiment}"
            )
        
        if sentiment > self.max_sentiment:
            return StreamValidationResult(
                is_valid=False,
                violation_severity=ViolationSeverity.MEDIUM,
                violation_message=f"Sentiment too positive: {sentiment:.2f} > {self.max_sentiment}"
            )
        
        return StreamValidationResult(is_valid=True)


class ToxicityDetector(StreamingContract):
    """Detects potentially toxic content in real-time."""
    
    def __init__(
        self,
        toxicity_threshold: float = 0.8,
        toxic_patterns: Optional[List[str]] = None,
        severity_escalation: bool = True,
    ):
        super().__init__(
            name="ToxicityDetector",
            description="Detects toxic content in real-time",
            validation_mode=ChunkValidationMode.IMMEDIATE
        )
        self.toxicity_threshold = toxicity_threshold
        self.severity_escalation = severity_escalation
        
        # Basic toxic patterns (in a real implementation, use ML models)
        default_toxic = [
            r'\b(hate|kill|die|murder)\b',
            r'\b(stupid|idiot|moron)\b',
            r'\b(f[*u]ck|sh[*i]t|damn)\b',
        ]
        
        self.toxic_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in (toxic_patterns or default_toxic)
        ]
        
        self.violation_count = 0
    
    async def validate_chunk(
        self,
        chunk: StreamChunk,
        context: Optional[Dict[str, Any]] = None
    ) -> StreamValidationResult:
        """Detect toxicity in real-time."""
        content = chunk.cumulative_content.lower()
        
        # Check for toxic patterns
        for pattern in self.toxic_patterns:
            matches = pattern.findall(content)
            if matches:
                self.violation_count += len(matches)
                
                # Escalate severity based on violation count
                if self.severity_escalation:
                    if self.violation_count >= 3:
                        severity = ViolationSeverity.CRITICAL
                        should_terminate = True
                    elif self.violation_count >= 2:
                        severity = ViolationSeverity.HIGH
                        should_terminate = True
                    else:
                        severity = ViolationSeverity.MEDIUM
                        should_terminate = False
                else:
                    severity = ViolationSeverity.HIGH
                    should_terminate = True
                
                return StreamValidationResult(
                    is_valid=False,
                    should_terminate=should_terminate,
                    violation_severity=severity,
                    violation_message=f"Toxic content detected: {pattern.pattern} (violations: {self.violation_count})",
                    auto_fix_suggestion="[TOXIC CONTENT REMOVED]"
                )
        
        return StreamValidationResult(is_valid=True)