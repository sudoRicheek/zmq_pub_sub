import time
import asyncio
from typing import Callable


class ConstantRateExecutor:
    """
    Utility class for executing async functions at a constant rate.
    """

    def __init__(self, rate_hz: float, func: Callable):
        """
        Initialize constant rate executor.

        Args:
            rate_hz: Target execution rate in Hz
            func: Async function to execute
        """
        self.func = func
        self.running = True
        self.interval = 1.0 / rate_hz

    async def run(self, *args, **kwargs) -> None:
        """
        Run a function at constant rate with precise timing.
        Executions happen at exact intervals regardless of execution time.

        Args:
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
        """
        next_exec_time = time.perf_counter()

        while self.running:
            current_time = time.perf_counter()

            if current_time >= next_exec_time:
                await self.func(*args, **kwargs)
                next_exec_time += self.interval

            await asyncio.sleep(
                min(
                    max(
                        0,
                        next_exec_time - time.perf_counter(),
                    ),
                    0.001,
                )
            )

    def stop(self):
        """Stop the executor."""
        self.running = False
