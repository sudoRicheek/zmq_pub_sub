"""
High-performance ZeroMQ Publisher for numpy arrays with zero-copy optimization.
Sends numpy arrays at a constant rate using asyncio for precise timing.
"""

import asyncio
import zmq
import zmq.asyncio

import time
import struct
import numpy as np

from utils import ConstantRateExecutor


class FastArrayPublisher:
    def __init__(
        self, ipc_path: str = "/tmp/0", send_rate_hz: float = 1000.0, stats_hz: float = 1.0
    ):
        """
        Initialize high-performance publisher.

        Args:
            ipc_path: IPC path to bind to
            send_rate_hz: Target sending rate in Hz
        """
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)

        # High-performance socket options
        self.socket.setsockopt(zmq.SNDHWM, 1000)  # High water mark
        self.socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue if no peers

        self.socket.bind(f"ipc://{ipc_path}")

        self.stats_hz = stats_hz
        self.send_rate_hz = send_rate_hz
        self.msg_counter = 0
        self.running = False

        self.header_struct = struct.Struct("!QIIII")  # counter, ndim, dtype, shape...

        # Statistics
        self.sent_count = 0
        self.failed_count = 0
        self.start_time = None
        self.array = None

        # Constant rate executor
        self._send_single_executor = ConstantRateExecutor(send_rate_hz, self._send_single)
        self._stats_executor = ConstantRateExecutor(stats_hz, self._stats_single)

        print(f"Publisher bound to ipc://{ipc_path}. Sending at {send_rate_hz} Hz")
        print(f"Send rate: {send_rate_hz} Hz, Stats rate: {stats_hz} Hz")

    async def send_array(self, array: np.ndarray, topic: str = "data") -> bool:
        """
        Send numpy array with zero-copy optimization.

        Args:
            array: numpy array to send
            topic: message topic

        Returns:
            True if sent successfully, False if would block
        """
        try:
            dtype_code = array.dtype.str.encode("ascii")
            shape_data = struct.pack("!" + "I" * len(array.shape), *array.shape)

            header = self.header_struct.pack(
                self.msg_counter, array.ndim, len(dtype_code), len(shape_data), array.nbytes
            )

            # Send multipart message: topic, header, dtype, shape, data
            await self.socket.send_multipart(
                [topic.encode("utf-8"), header, dtype_code, shape_data, array],
                flags=zmq.NOBLOCK,
                copy=True, # sometimes setting copy=True, might be necessary. Check your use-case
            )

            self.msg_counter += 1
            return True

        except zmq.Again:
            return False

    async def _send_single(self):
        """
        Send a single array. Used by the constant rate executor.
        """
        # Update array data (simulate changing data)
        self.array += 0.1

        if await self.send_array(self.array):
            self.sent_count += 1
        else:
            self.failed_count += 1

    async def _stats_single(self):
        """
        Async loop for printing statistics.
        """
        elapsed = time.perf_counter() - self.start_time

        if self.sent_count > 0:
            rate = self.sent_count / elapsed
            target_rate = self.send_rate_hz
            accuracy = (rate / target_rate) * 100 if target_rate > 0 else 0

            print(
                f"Sent: {self.sent_count}, Failed: {self.failed_count}, "
                f"Rate: {rate:.6f} Hz (target: {target_rate:.1f}, accuracy: {accuracy:.1f}%), "
                f"Elapsed: {elapsed:.1f}s"
            )

    async def run_constant_rate(
        self,
        array_shape: tuple = (1000, 1000),
        dtype: np.dtype = np.float32,
    ):
        """
        Run publisher at constant rate using asyncio.

        Args:
            array_shape: Shape of arrays to generate
            dtype: Data type of arrays
            stats_interval: How often to print stats in seconds
        """
        print(f"Starting asyncio-based constant rate transmission...")
        print(f"Array shape: {array_shape}, dtype: {dtype}")
        print(f"Array size: {np.prod(array_shape) * dtype().itemsize / 1024 / 1024:.2f} MB")

        # Pre-allocate array for reuse
        self.array = np.random.random(array_shape).astype(dtype)

        self.start_time = time.perf_counter()
        self.sent_count = 0
        self.failed_count = 0
        self.running = True

        # Create concurrent tasks
        send_task = asyncio.create_task(self._send_single_executor.run())
        stats_task = asyncio.create_task(self._stats_executor.run())

        tasks = [send_task, stats_task]

        try:
            print("Starting asyncio event loop...")
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            print("\nCancelling publisher tasks...")
        finally:
            self.running = False
            self._send_single_executor.stop()
            self._stats_executor.stop()

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - self.start_time
        actual_rate = self.sent_count / elapsed if elapsed > 0 else 0
        target_rate = self.send_rate_hz

        print(f"\nFinal stats:")
        print(f"Sent: {self.sent_count}, Failed: {self.failed_count}")
        print(f"Target rate: {target_rate:.1f} Hz")
        print(f"Actual rate: {actual_rate:.1f} Hz")
        print(f"Rate accuracy: {(actual_rate/target_rate)*100:.2f}%")
        print(f"Total time: {elapsed:.2f}s")

    async def close(self):
        """Clean shutdown."""
        self.running = False
        self._send_single_executor.stop()
        self._stats_executor.stop()
        print("Closing publisher...")
        self.socket.close()
        self.context.term()


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="High-performance asyncio ZMQ numpy array publisher"
    )
    parser.add_argument("--ipc-path", type=str, default="/tmp/0", help="IPC path to bind to")
    parser.add_argument("--rate", type=float, default=1000.0, help="Send rate in Hz")
    parser.add_argument(
        "--shape", type=str, default="1000,1000", help="Array shape (comma-separated)"
    )
    parser.add_argument("--dtype", type=str, default="float32", help="Array data type")
    parser.add_argument(
        "--stats-hz", type=float, default=1.0, help="Stats printing interval in seconds"
    )

    args = parser.parse_args()

    shape = tuple(map(int, args.shape.split(",")))
    dtype = getattr(np, args.dtype)

    publisher = FastArrayPublisher(args.ipc_path, args.rate)

    try:
        await publisher.run_constant_rate(shape, dtype)
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        await publisher.close()


if __name__ == "__main__":
    asyncio.run(main())
