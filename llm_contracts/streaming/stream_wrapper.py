"""Stream wrappers for real-time validation integration."""

import asyncio
import json
import time
from typing import Any, AsyncIterator, Iterator, Optional, Dict, List, Union
import logging

from .stream_validator import StreamingValidator, StreamValidationResult
from ..contracts.base import ContractBase
from ..core.exceptions import ContractViolationError

logger = logging.getLogger(__name__)


class StreamWrapper:
    """Synchronous stream wrapper with real-time validation."""
    
    def __init__(
        self,
        original_stream: Iterator[Any],
        validator: StreamingValidator,
        auto_fix: bool = True,
        raise_on_violation: bool = True,
    ):
        self.original_stream = original_stream
        self.validator = validator
        self.auto_fix = auto_fix
        self.raise_on_violation = raise_on_violation
        self._chunk_index = 0
        self._terminated = False
        self._final_result: Optional[StreamValidationResult] = None
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._terminated:
            raise StopIteration
        
        try:
            chunk = next(self.original_stream)
            
            # Extract content from OpenAI chunk
            content = self._extract_content(chunk)
            
            if content:
                # Run synchronous validation (note: this blocks)
                # In production, consider using a separate thread for validation
                validation_result = asyncio.run(
                    self.validator.process_chunk(content, self._chunk_index)
                )
                
                self._chunk_index += 1
                
                # Handle validation results
                if not validation_result.is_valid:
                    if validation_result.should_terminate:
                        self._terminated = True
                        logger.warning(f"Stream terminated due to violation: {validation_result.violation_message}")
                        
                        if self.raise_on_violation:
                            raise ContractViolationError(
                                f"Critical violation in stream: {validation_result.violation_message}"
                            )
                        else:
                            raise StopIteration
                    
                    elif self.raise_on_violation:
                        raise ContractViolationError(
                            f"Validation failed: {validation_result.violation_message}"
                        )
                    else:
                        logger.warning(f"Validation warning: {validation_result.violation_message}")
                
                # Apply auto-fix if available
                if (not validation_result.is_valid and 
                    self.auto_fix and 
                    validation_result.auto_fix_suggestion):
                    # Modify chunk content (implementation depends on chunk structure)
                    chunk = self._apply_auto_fix(chunk, validation_result.auto_fix_suggestion)
            
            return chunk
            
        except StopIteration:
            # Stream ended - perform final validation
            if not self._terminated:
                self._final_result = asyncio.run(self.validator.finalize_validation())
                
                if (not self._final_result.is_valid and 
                    self.raise_on_violation):
                    raise ContractViolationError(
                        f"Final validation failed: {self._final_result.violation_message}"
                    )
            
            raise
        
        except Exception as e:
            logger.error(f"Error in stream validation: {e}")
            if self.raise_on_violation:
                raise
            return chunk
    
    def _extract_content(self, chunk: Any) -> str:
        """Extract text content from OpenAI streaming chunk."""
        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    return choice.delta.content or ""
                elif hasattr(choice, 'text'):
                    return choice.text or ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract content from chunk: {e}")
            return ""
    
    def _apply_auto_fix(self, chunk: Any, auto_fix: str) -> Any:
        """Apply auto-fix to chunk (basic implementation)."""
        # This is a simplified implementation
        # In practice, you'd need to modify the chunk structure appropriately
        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    choice.delta.content = auto_fix
        except Exception as e:
            logger.warning(f"Failed to apply auto-fix: {e}")
        
        return chunk
    
    def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        metrics = self.validator.get_metrics().get_summary()
        if self._final_result:
            metrics.update(self._final_result.performance_metrics)
        return metrics


class AsyncStreamWrapper:
    """Asynchronous stream wrapper with real-time validation."""
    
    def __init__(
        self,
        original_stream: AsyncIterator[Any],
        validator: StreamingValidator,
        auto_fix: bool = True,
        raise_on_violation: bool = True,
        concurrent_validation: bool = True,
    ):
        self.original_stream = original_stream
        self.validator = validator
        self.auto_fix = auto_fix
        self.raise_on_violation = raise_on_violation
        self.concurrent_validation = concurrent_validation
        self._chunk_index = 0
        self._terminated = False
        self._final_result: Optional[StreamValidationResult] = None
        self._validation_tasks: List[asyncio.Task] = []
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self._terminated:
            raise StopAsyncIteration
        
        try:
            chunk = await self.original_stream.__anext__()
            
            # Extract content from chunk
            content = self._extract_content(chunk)
            
            if content:
                if self.concurrent_validation:
                    # Run validation concurrently (non-blocking)
                    validation_task = asyncio.create_task(
                        self._validate_chunk_async(content, self._chunk_index)
                    )
                    self._validation_tasks.append(validation_task)
                    
                    # Check if any previous validations failed
                    await self._check_completed_validations()
                else:
                    # Run validation synchronously (blocking)
                    validation_result = await self.validator.process_chunk(
                        content, self._chunk_index
                    )
                    await self._handle_validation_result(validation_result, chunk)
                
                self._chunk_index += 1
            
            return chunk
            
        except StopAsyncIteration:
            # Stream ended - wait for all validations to complete
            if self._validation_tasks:
                await asyncio.gather(*self._validation_tasks, return_exceptions=True)
            
            # Perform final validation
            if not self._terminated:
                self._final_result = await self.validator.finalize_validation()
                
                if (not self._final_result.is_valid and 
                    self.raise_on_violation):
                    raise ContractViolationError(
                        f"Final validation failed: {self._final_result.violation_message}"
                    )
            
            raise
        
        except Exception as e:
            logger.error(f"Error in async stream validation: {e}")
            if self.raise_on_violation:
                raise
            return chunk
    
    async def _validate_chunk_async(self, content: str, chunk_index: int) -> StreamValidationResult:
        """Validate chunk asynchronously."""
        try:
            return await self.validator.process_chunk(content, chunk_index)
        except Exception as e:
            logger.error(f"Async validation error for chunk {chunk_index}: {e}")
            return StreamValidationResult(
                is_valid=False,
                violation_message=f"Validation error: {str(e)}"
            )
    
    async def _check_completed_validations(self):
        """Check completed validation tasks."""
        completed_tasks = [task for task in self._validation_tasks if task.done()]
        
        for task in completed_tasks:
            self._validation_tasks.remove(task)
            
            try:
                validation_result = task.result()
                await self._handle_validation_result(validation_result, None)
            except Exception as e:
                logger.error(f"Error handling completed validation: {e}")
                if self.raise_on_violation:
                    raise
    
    async def _handle_validation_result(
        self,
        validation_result: StreamValidationResult,
        chunk: Optional[Any]
    ):
        """Handle validation result."""
        if not validation_result.is_valid:
            if validation_result.should_terminate:
                self._terminated = True
                logger.warning(f"Stream terminated due to violation: {validation_result.violation_message}")
                
                if self.raise_on_violation:
                    raise ContractViolationError(
                        f"Critical violation in stream: {validation_result.violation_message}"
                    )
            
            elif self.raise_on_violation:
                raise ContractViolationError(
                    f"Validation failed: {validation_result.violation_message}"
                )
            else:
                logger.warning(f"Validation warning: {validation_result.violation_message}")
    
    def _extract_content(self, chunk: Any) -> str:
        """Extract text content from streaming chunk."""
        try:
            if hasattr(chunk, 'choices') and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                    return choice.delta.content or ""
                elif hasattr(choice, 'text'):
                    return choice.text or ""
            return ""
        except Exception as e:
            logger.warning(f"Failed to extract content from chunk: {e}")
            return ""
    
    async def get_validation_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        # Wait for pending validations
        if self._validation_tasks:
            await asyncio.gather(*self._validation_tasks, return_exceptions=True)
        
        metrics = self.validator.get_metrics().get_summary()
        if self._final_result:
            metrics.update(self._final_result.performance_metrics)
        return metrics
    
    async def cleanup(self):
        """Clean up resources."""
        # Cancel pending validation tasks
        for task in self._validation_tasks:
            if not task.done():
                task.cancel()
        
        if self._validation_tasks:
            await asyncio.gather(*self._validation_tasks, return_exceptions=True)
        
        await self.validator.cleanup()


class StreamingClient:
    """Enhanced OpenAI client with streaming validation."""
    
    def __init__(
        self,
        base_client: Any,
        contracts: Optional[List[ContractBase]] = None,
        default_validation_mode: str = "adaptive",
        enable_performance_monitoring: bool = True,
        max_concurrent_validations: int = 10,
    ):
        self.base_client = base_client
        self.contracts = contracts or []
        self.default_validation_mode = default_validation_mode
        self.enable_performance_monitoring = enable_performance_monitoring
        self.max_concurrent_validations = max_concurrent_validations
        
        # Validation semaphore to limit concurrent validations
        self.validation_semaphore = asyncio.Semaphore(max_concurrent_validations)
    
    def add_contract(self, contract: ContractBase):
        """Add a validation contract."""
        self.contracts.append(contract)
    
    def create_streaming_validator(self) -> StreamingValidator:
        """Create a streaming validator with current contracts."""
        from .stream_validator import ChunkValidationMode
        
        mode_map = {
            "immediate": ChunkValidationMode.IMMEDIATE,
            "buffered": ChunkValidationMode.BUFFERED,
            "threshold": ChunkValidationMode.THRESHOLD,
            "adaptive": ChunkValidationMode.ADAPTIVE,
        }
        
        return StreamingValidator(
            contracts=self.contracts,
            validation_mode=mode_map.get(self.default_validation_mode, ChunkValidationMode.ADAPTIVE),
            performance_monitoring=self.enable_performance_monitoring,
            early_termination=True,
        )
    
    async def chat_completions_create_stream(self, **kwargs) -> AsyncStreamWrapper:
        """Create streaming chat completion with validation."""
        # Force streaming
        kwargs['stream'] = True
        
        # Create original stream
        original_stream = await self.base_client.chat.completions.create(**kwargs)
        
        # Create validator
        validator = self.create_streaming_validator()
        
        # Wrap with validation
        return AsyncStreamWrapper(
            original_stream=original_stream,
            validator=validator,
            concurrent_validation=True,
        )
    
    def chat_completions_create_stream_sync(self, **kwargs) -> StreamWrapper:
        """Create synchronous streaming chat completion with validation."""
        # Force streaming
        kwargs['stream'] = True
        
        # Create original stream
        original_stream = self.base_client.chat.completions.create(**kwargs)
        
        # Create validator
        validator = self.create_streaming_validator()
        
        # Wrap with validation
        return StreamWrapper(
            original_stream=original_stream,
            validator=validator,
        )
    
    async def completions_create_stream(self, **kwargs) -> AsyncStreamWrapper:
        """Create streaming completion with validation."""
        # Force streaming
        kwargs['stream'] = True
        
        # Create original stream
        original_stream = await self.base_client.completions.create(**kwargs)
        
        # Create validator
        validator = self.create_streaming_validator()
        
        # Wrap with validation
        return AsyncStreamWrapper(
            original_stream=original_stream,
            validator=validator,
            concurrent_validation=True,
        )
    
    def __getattr__(self, name: str) -> Any:
        """Delegate to base client for non-streaming methods."""
        return getattr(self.base_client, name)


# Convenience functions for quick setup

def create_streaming_validator_from_patterns(
    forbidden_patterns: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    require_json: bool = False,
    toxicity_detection: bool = True,
) -> StreamingValidator:
    """Create a streaming validator with common patterns."""
    from .stream_processors import ContentFilter, LengthMonitor, JSONValidator, ToxicityDetector
    
    contracts = []
    
    if forbidden_patterns:
        contracts.append(ContentFilter(
            blocked_patterns=forbidden_patterns,
            stop_on_violation=True,
        ))
    
    if max_length:
        contracts.append(LengthMonitor(
            max_length=max_length,
            truncate_on_limit=True,
        ))
    
    if require_json:
        contracts.append(JSONValidator(
            require_complete_json=True,
        ))
    
    if toxicity_detection:
        contracts.append(ToxicityDetector(
            severity_escalation=True,
        ))
    
    return StreamingValidator(
        contracts=contracts,
        early_termination=True,
        performance_monitoring=True,
    )


async def validate_stream(
    stream: Union[Iterator[Any], AsyncIterator[Any]],
    contracts: List[ContractBase],
    **kwargs
) -> Union[StreamWrapper, AsyncStreamWrapper]:
    """Convenience function to add validation to any stream."""
    validator = StreamingValidator(
        contracts=contracts,
        **kwargs
    )
    
    if hasattr(stream, '__aiter__'):
        return AsyncStreamWrapper(stream, validator)
    else:
        return StreamWrapper(stream, validator)