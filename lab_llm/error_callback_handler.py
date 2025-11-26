import logging
from typing import Any, Optional, Union
from langchain.callbacks.base import BaseCallbackHandler

from lab_llm.error_tracker import ErrorCategory, ErrorTracker


class ErrorCallbackHandler(BaseCallbackHandler):
    """
    Enhanced error callback handler with error classification and tracking.

    Provides:
    - Error classification (transient/permanent/user_interrupt)
    - Optional JSONL-based error logging for analysis
    - Structured logging with error categories
    - User interrupt propagation
    """

    def __init__(
        self,
        logger: logging.Logger,
        error_tracker: Optional[ErrorTracker] = None,
        propagate_interrupts: bool = True,
    ):
        """
        Initialize error callback handler.

        Args:
            logger: Python logger for console/file logging
            error_tracker: Optional ErrorTracker for JSONL logging
            propagate_interrupts: If True, re-raise KeyboardInterrupt (default: True)
        """
        super().__init__()
        self.logger = logger
        self.error_tracker = error_tracker
        self.propagate_interrupts = propagate_interrupts

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any
    ) -> Optional[ErrorCategory]:
        """
        Handle LLM errors with classification and tracking.

        Args:
            error: The exception that occurred
            **kwargs: Additional context from LangChain

        Returns:
            ErrorCategory if error_tracker is enabled, None otherwise

        Raises:
            KeyboardInterrupt: If propagate_interrupts=True and error is KeyboardInterrupt
        """
        # Classify the error
        category = None
        if self.error_tracker:
            category = self.error_tracker.classify_error(error)

            # Extract context from kwargs if available
            context = {}
            if 'run_id' in kwargs:
                context['run_id'] = str(kwargs['run_id'])

            # Log to ErrorTracker
            # Note: prompt and prompt_hash will be added by the calling code
            # (from _run_batch) if available
            self.error_tracker.log_error(
                error=error,
                context=context,
                include_traceback=False,  # Keep logs lightweight
            )
        else:
            # Basic classification without tracker
            if isinstance(error, KeyboardInterrupt):
                category = ErrorCategory.USER_INTERRUPT

        # Enhanced structured logging with category
        if category:
            self.logger.error(f"LLM Error [{category.value}]: {type(error).__name__}")
        else:
            self.logger.error(f"LLM Error: {type(error).__name__}")

        self.logger.error(f"Message: {str(error)}")
        self.logger.error(f"Additional context: {kwargs}")

        # Propagate user interrupts to stop execution
        if self.propagate_interrupts and isinstance(error, KeyboardInterrupt):
            self.logger.warning("User interrupt detected - stopping execution")
            raise error

        return category

