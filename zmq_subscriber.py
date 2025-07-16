"""
High-performance ZeroMQ Subscriber for numpy arrays with true zero-copy reception.
Uses ZMQ's zero-copy message API for maximum performance with async/await.
"""

import asyncio
import zmq
import zmq.asyncio

import time
import struct
import argparse
import numpy as np
from typing import Optional, Tuple

from utils import ConstantRateExecutor


class FastArraySubscriber:
    def __init__(
        self, ipc_path: str = "/tmp/0", process_rate_hz: float = 1000.0, stats_hz: float = 1.0
    ):
        """
        Initialize zero-copy subscriber.

        Args:
            host: ZMQ host to connect to
            port: ZMQ port to connect to
        """
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.SUB)

        # High-performance socket options
        self.socket.setsockopt(zmq.RCVHWM, 1000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVBUF, 100)  # Use ZMQ buffering

        self.socket.connect(f"ipc://{ipc_path}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "data")

        self.stats_hz = stats_hz
        self.process_rate_hz = process_rate_hz
        self.header_struct = struct.Struct("!QIIII")
        self.running = False

        self._process_executor = ConstantRateExecutor(process_rate_hz, self._process_single)
        self._stats_executor = ConstantRateExecutor(stats_hz, self._stats_single)

        print(f"Subscriber connected to ipc://{ipc_path}")

    def _create_array_from_zmq_message(
        self, msg: zmq.Message, dtype: np.dtype, shape: tuple
    ) -> np.ndarray:
        """
        Create numpy array from ZMQ message with zero-copy.

        Args:
            msg: ZMQ Message object
            dtype: numpy data type
            shape: array shape

        Returns:
            numpy array that shares buffer with ZMQ message
        """
        return np.frombuffer(msg.buffer, dtype=dtype).reshape(shape)

    async def receive_array(self) -> Optional[Tuple[np.ndarray, int, zmq.Message]]:
        """
        Receive numpy array with true zero-copy.

        Returns:
            Tuple of (array, message_counter, zmq_message) or None
            Note: Keep zmq_message alive as long as you need the array!
        """
        try:
            messages = await self.socket.recv_multipart(zmq.NOBLOCK, copy=False)

            if len(messages) != 5:
                print(f"Warning: Expected 5 parts, got {len(messages)}")
                return None

            topic_msg, header_msg, dtype_msg, shape_msg, array_msg = messages

            header = self.header_struct.unpack(header_msg.buffer)
            msg_counter, ndim, dtype_len, shape_len, nbytes = header

            dtype_str = dtype_msg.buffer.tobytes().decode("ascii")
            dtype = np.dtype(dtype_str)

            shape = struct.unpack("!" + "I" * ndim, shape_msg.buffer)
            array = self._create_array_from_zmq_message(array_msg, dtype, shape)

            return array, msg_counter, array_msg

        except zmq.Again:
            return None
        except Exception as e:
            print(f"Error receiving array: {e}")
            return None

    async def run_realtime_mode(self):
        """
        Run subscriber in real-time mode - process latest available data.
        """
        print("Starting async real-time mode reception...")

        start_time = time.perf_counter()

        stats = {
            "received_count": 0,
            "processed_count": 0,
            "dropped_messages": 0,
            "last_msg_counter": -1,
        }

        self.running = True

        receive_task = asyncio.create_task(self._receive_loop(stats))
        process_task = asyncio.create_task(self._process_executor.run(stats))
        stats_task = asyncio.create_task(self._stats_executor.run(start_time, stats))

        tasks = [receive_task, process_task, stats_task]
        try:
            # Run all tasks concurrently
            await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            print("\nCancelling subscriber tasks...")
        finally:
            self.running = False
            self._process_executor.stop()
            self._stats_executor.stop()

            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start_time
        print(f"\nFinal stats:")
        print(f"Received: {stats['received_count']}, Processed: {stats['processed_count']}")
        print(f"Dropped messages: {stats['dropped_messages']}")
        print(f"Average recv rate: {stats['received_count'] / elapsed:.1f} Hz")
        print(f"Average proc rate: {stats['processed_count'] / elapsed:.1f} Hz")
        print(f"Total time: {elapsed:.2f}s")

    async def _receive_loop(self, stats: dict):
        """Internal receive loop - runs as fast as possible."""
        self.latest_msg_ref = None
        self.latest_array = None
        self.latest_counter = -1

        while self.running:
            try:
                result = await self.receive_array()

                if result is not None:
                    array, msg_counter, msg_ref = result

                    self.latest_msg_ref = msg_ref  # Keep message alive
                    self.latest_array = array
                    self.latest_counter = msg_counter
                    stats["received_count"] += 1

                    if (
                        stats["last_msg_counter"] >= 0
                        and msg_counter > stats["last_msg_counter"] + 1
                    ):
                        stats["dropped_messages"] += msg_counter - stats["last_msg_counter"] - 1

                    stats["last_msg_counter"] = msg_counter

                await asyncio.sleep(0)
            except Exception as e:
                print(f"Error in receive loop: {e}")
                await asyncio.sleep(0.001)

    async def _stats_single(self, start_time: float, stats: dict):
        """Internal stats reporting loop."""
        elapsed = time.perf_counter() - start_time
        recv_rate = stats["received_count"] / elapsed if elapsed > 0 else 0
        proc_rate = stats["processed_count"] / elapsed if elapsed > 0 else 0

        if hasattr(self, "latest_array") and self.latest_array is not None:
            array_size_mb = self.latest_array.nbytes / 1024 / 1024
            throughput_mbps = (stats["received_count"] * array_size_mb * 8) / elapsed / 1000

            print(
                f"Received: {stats['received_count']}, Processed: {stats['processed_count']}, "
                f"Dropped: {stats['dropped_messages']}, "
                f"Recv Rate: {recv_rate:.1f} Hz, "
                f"Proc Rate: {proc_rate:.1f} Hz, "
                f"Throughput: {throughput_mbps:.1f} Mbps"
            )

    async def _process_single(self, stats: dict):
        """Internal process loop"""
        if hasattr(self, "latest_array") and self.latest_array is not None:
            await self.process_array(self.latest_array, self.latest_counter)
            stats["processed_count"] += 1

    async def process_array(self, array: np.ndarray, msg_counter: int):
        """
        Process single received array (zero-copy).

        Args:
            array: Received numpy array (shares memory with ZMQ message)
            msg_counter: Message counter
        """
        if msg_counter % 100 == 0:
            mean_val = np.mean(array)
            std_val = np.std(array)
            print(
                f"MSG {msg_counter}: shape={array.shape}, mean={mean_val:.3f}, "
                f"std={std_val:.3f}, size={array.nbytes/1024/1024:.1f}MB"
            )

    async def close(self):
        """Clean shutdown."""
        self.running = False
        self._process_executor.stop()
        self._stats_executor.stop()
        print("Closing subscriber...")
        self.socket.close()
        self.context.term()


async def main():
    parser = argparse.ArgumentParser(description="Zero-copy ZMQ numpy array subscriber")
    parser.add_argument("--ipc-path", type=str, default="/tmp/0", help="IPC path to connect to")
    parser.add_argument("--process-rate-hz", type=float, default=1000.0, help="Process rate in Hz")
    parser.add_argument("--stats-hz", type=float, default=1.0, help="Stats hz in hz")

    args = parser.parse_args()

    subscriber = FastArraySubscriber(args.ipc_path, args.process_rate_hz, args.stats_hz)

    try:
        await subscriber.run_realtime_mode()
    except KeyboardInterrupt:
        print("\nStopping subscriber...")
    finally:
        await subscriber.close()


if __name__ == "__main__":
    asyncio.run(main())
