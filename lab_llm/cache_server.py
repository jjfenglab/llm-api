"""
Cache server for managing DuckDB connections in a multi-process environment
"""

import os
import pickle
import random
import socket
import subprocess
import sys
import threading
import time
from enum import Enum
from typing import List, Union

import msgpack

from lab_llm.constants import LLMModel, convert_to_llm_type
from lab_llm.duckdb_handler import DuckDBHandler
from lab_llm.llm_cache import LLMCache


def serialize_for_msgpack(obj):
    """Custom serializer to handle LLMModel and other complex objects"""
    if isinstance(obj, LLMModel):
        return {"__llm_model__": obj.name.value}
    elif isinstance(obj, Enum):
        return {"__enum__": {"type": obj.__class__.__name__, "value": obj.value}}
    return obj


def deserialize_from_msgpack(obj):
    """Custom deserializer to handle LLMModel and other complex objects"""
    if isinstance(obj, dict):
        if "__llm_model__" in obj:
            # Reconstruct LLMModel from string value
            return convert_to_llm_type(obj["__llm_model__"])
        elif "__enum__" in obj:
            # Handle other enum types if needed
            return obj["__enum__"]["value"]
    return obj


def msgpack_encode(data):
    """Encode data with custom object handling"""
    try:
        result = msgpack.packb(data, default=serialize_for_msgpack)
        # Check if result is too large (>200KB), fallback to pickle
        if len(result) > 200000:
            # Prepend a magic byte to indicate pickle format
            return b"\x01" + pickle.dumps(data)
        return b"\x00" + result  # Prepend magic byte for msgpack
    except Exception as e:
        return b"\x01" + pickle.dumps(data)


def msgpack_decode(data):
    """Decode data with custom object handling"""
    if len(data) == 0:
        raise ValueError("Empty data received")

    # Check magic byte
    if data[0] == 1:  # Pickle format
        return pickle.loads(data[1:])
    elif data[0] == 0:  # msgpack format
        return msgpack.unpackb(
            data[1:], object_hook=deserialize_from_msgpack, raw=False
        )
    else:
        # Legacy format (no magic byte) - assume msgpack
        return msgpack.unpackb(data, object_hook=deserialize_from_msgpack, raw=False)


class CacheServer:
    def __init__(
        self,
        db_path: str,
        port: int = 9999,
        chunk_size: int = None,
        socket_buffer_size: int = None,
    ):
        try:
            self.db_path = db_path
            self.port = port
            self.socket_path = f"/tmp/duckdb_cache_{port}.sock"

            # Dynamic chunk sizing - larger chunks for local communication
            self.chunk_size = chunk_size or self._get_optimal_chunk_size()
            self.socket_buffer_size = socket_buffer_size or (
                128 * 1024
            )  # 128KB default

            print(f"Initializing DuckDB handler for {db_path}")
            self.handler = DuckDBHandler(db_path, read_only=False)
            print("Creating LLM cache")
            self.cache = LLMCache(self.handler)
            self.lock = threading.Lock()
            self.running = False
            print(
                f"Cache server initialized successfully with chunk_size={self.chunk_size}, buffer_size={self.socket_buffer_size}"
            )
        except Exception as e:
            print(f"Error initializing cache server: {e}")
            raise

    def _get_optimal_chunk_size(self):
        # 64 KB seems to work ok for local communication
        return 64 * 1024  # 64KB

    def _get_dynamic_chunk_size(self, data_size):
        """Get chunk size based on the total data size"""
        if data_size <= 8 * 1024:  # <= 8KB
            return min(self.chunk_size, 4096)
        elif data_size <= 64 * 1024:  # <= 64KB
            return min(self.chunk_size, 16 * 1024)  # 16KB chunks
        else:  # > 64KB
            return self.chunk_size  # Use full chunk size for large data

    def handle_request(self, data):
        """Handle cache requests from clients"""
        try:
            request = msgpack_decode(data)
            operation = request["operation"]

            with self.lock:  # Ensure thread safety
                if operation == "get_response":
                    result = self.cache.get_response(
                        text=request["prompt"],
                        model_type=request["model_type"],
                        seed=request["seed"],
                        max_new_tokens=request["max_new_tokens"],
                        temperature=request["temperature"],
                    )
                    return {"status": "success", "data": result}

                elif operation == "save_response":
                    self.cache.save_response(
                        input_text=request["prompt"],
                        llm_output=request["llm_output"],
                        model_type=request["model_type"],
                        seed=request["seed"],
                        max_new_tokens=request["max_new_tokens"],
                        temperature=request["temperature"],
                    )
                    return {"status": "success", "data": None}

                elif operation == "get_responses":
                    result = self.cache.get_responses(
                        texts=request["batch_prompts"],
                        model_type=request["model_type"],
                        seed=request["seed"],
                        max_new_tokens=request["max_new_tokens"],
                        temperature=request["temperature"],
                    )
                    return {"status": "success", "data": result}

                elif operation == "save_responses":
                    self.cache.save_responses(
                        input_texts=request["batch_prompts"],
                        llm_outputs=request["llm_outputs"],
                        model_type=request["model_type"],
                        seed=request["seed"],
                        max_new_tokens=request["max_new_tokens"],
                        temperature=request["temperature"],
                    )
                    return {"status": "success", "data": None}

                else:
                    return {
                        "status": "error",
                        "message": f"Unknown operation: {operation}",
                    }

        except Exception as e:
            print(f"DEBUG SERVER: Error in handle_request: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def start(self):
        """Start the cache server"""
        try:
            self.running = True
            # Remove old socket file if it exists
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

            server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server_socket.bind(self.socket_path)
            server_socket.listen(5)

            print(f"Cache server started on Unix socket: {self.socket_path}")
            print(f"Database: {self.db_path}")
        except Exception as e:
            print(f"Error starting cache server: {e}")
            raise

        try:
            while self.running:
                try:
                    client_socket, address = server_socket.accept()
                    thread = threading.Thread(
                        target=self.handle_client, args=(client_socket, address)
                    )
                    thread.start()
                except Exception as e:
                    if (
                        self.running
                    ):  # Only log errors if we're still supposed to be running
                        print(f"Error accepting connection: {e}")
        finally:
            server_socket.close()
            self.handler.close_connection()
            # Clean up socket file
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)

    def handle_client(self, client_socket, address):
        """Handle individual client connections"""
        try:

            client_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, self.socket_buffer_size
            )
            client_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, self.socket_buffer_size
            )

            # Receive data size first
            size_data = client_socket.recv(4)
            if not size_data:
                return

            data_size = int.from_bytes(size_data, byteorder="big")

            # Get optimal chunk size for this data size
            chunk_size = self._get_dynamic_chunk_size(data_size)

            # Receive the actual data
            data = b""
            while len(data) < data_size:
                remaining = data_size - len(data)
                chunk = client_socket.recv(min(chunk_size, remaining))
                if not chunk:
                    break
                data += chunk

            if len(data) == data_size:
                response = self.handle_request(data)
                response_data = msgpack_encode(response)

                # Send response size first, then data
                size_bytes = len(response_data).to_bytes(4, byteorder="big")
                client_socket.sendall(size_bytes)  # sendall ensures all data is sent
                client_socket.sendall(response_data)  # sendall ensures all data is sent

        except Exception as e:
            print(f"Error handling client {address}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            client_socket.close()

    def stop(self):
        """Stop the cache server"""
        self.running = False


class CacheClient:
    def __init__(
        self,
        port: int = None,
        cache_file: str = None,
        chunk_size: int = None,
        socket_buffer_size: int = None,
    ):
        self.cache_file = cache_file
        self.server_process = None
        self._auto_started_server = False

        # Dynamic chunk sizing - larger chunks for local communication
        self.chunk_size = chunk_size or (64 * 1024)  # 64KB default
        self.socket_buffer_size = socket_buffer_size or (128 * 1024)  # 128KB default

        # Find available port if not specified and cache_file is provided
        if cache_file and type(port) is not int:
            self.port = self._find_available_port()
        else:
            self.port = port or 9999

        self.socket_path = f"/tmp/duckdb_cache_{self.port}.sock"

        # If cache_file is provided, auto-start server if needed
        if cache_file:
            self._ensure_server_running()

        time.sleep(
            random.uniform(0.1, 0.5)
        )  # sleep random amount to avoid all processes trying to start server at the same time

    def _get_dynamic_chunk_size(self, data_size):
        """Get chunk size based on the total data size"""
        if data_size <= 8 * 1024:  # <= 8KB
            return min(self.chunk_size, 4096)  # Use smaller chunks for small data
        elif data_size <= 64 * 1024:  # <= 64KB
            return min(self.chunk_size, 16 * 1024)  # 16KB chunks
        else:  # > 64KB
            return self.chunk_size  # Use full chunk size for large data

    def _find_available_port(self, start_port: int = 9999):
        """Find an available port starting from start_port"""
        import socket

        for port in range(start_port, start_port + 10):  # Try 10 ports
            socket_path = f"/tmp/duckdb_cache_{port}.sock"
            # Check if socket file exists
            if os.path.exists(socket_path):
                # Try to connect to see if server is actually running
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                        sock.settimeout(1)
                        sock.connect(socket_path)
                        continue  # Server is running, try next port
                except:
                    # Socket file exists but no server, remove stale file
                    os.unlink(socket_path)
                    return port
            else:
                # No socket file, port is available
                return port
        raise Exception(f"No available ports found starting from {start_port}")

    def _is_server_running(self):
        """Check if cache server is already running on this socket"""
        try:
            import socket

            if not os.path.exists(self.socket_path):
                return False

            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(self.socket_path)
                return True
        except:
            # Clean up stale socket file
            if os.path.exists(self.socket_path):
                try:
                    os.unlink(self.socket_path)
                except:
                    pass
            return False

    def _ensure_server_running(self):
        """Start cache server if not already running"""
        if self._is_server_running():
            print(f"Cache server already running on port {self.port}")
            return

        print(
            f"Starting cache server for {self.cache_file} on socket {self.socket_path}"
        )

        # Start server process
        import subprocess

        server_cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "cache_server.py"),
            self.cache_file,
            str(self.port),
        ]

        self.server_process = subprocess.Popen(
            server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd(),  # Ensure the server starts with the right working directory
        )
        self._auto_started_server = True

        # Wait for server to start
        import time

        for attempt in range(10):  # 5 second timeout
            if self._is_server_running():
                print("Cache server started successfully")
                break
            time.sleep(0.5)

            # Check if process has exited
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                error_msg = f"Cache server process exited early (exit code: {self.server_process.returncode})"
                if stderr:
                    error_msg += f"\nStderr: {stderr}"
                if stdout:
                    error_msg += f"\nStdout: {stdout}"
                raise Exception(error_msg)
        else:
            # Process is still running but server not responding - get any output
            try:
                stdout, stderr = self.server_process.communicate(timeout=1)
                error_details = ""
                if stderr:
                    error_details += f"\nStderr: {stderr}"
                if stdout:
                    error_details += f"\nStdout: {stdout}"
            except subprocess.TimeoutExpired:
                error_details = "\nProcess is still running but not responding"

            raise Exception(
                f"Cache server failed to start within 5 seconds{error_details}"
            )

    def close(self):
        """Clean up - stop server if we started it"""
        if self._auto_started_server and self.server_process:
            print("Stopping auto-started cache server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                self.server_process.wait()
            self._auto_started_server = False

    def __del__(self):
        """Cleanup on deletion"""
        self.close()

    def _send_request(self, request):
        """Send request to cache server"""
        try:
            client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

            client_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_RCVBUF, self.socket_buffer_size
            )
            client_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, self.socket_buffer_size
            )

            client_socket.connect(self.socket_path)

            # Serialize request
            request_data = msgpack_encode(request)

            # Send data size first, then data
            client_socket.sendall(len(request_data).to_bytes(4, byteorder="big"))
            client_socket.sendall(request_data)

            # Receive response size first
            size_data = client_socket.recv(4)
            if len(size_data) != 4:
                raise Exception(f"Expected 4 bytes for size, got {len(size_data)}")

            response_size = int.from_bytes(size_data, byteorder="big")

            # Get optimal chunk size for this response size
            chunk_size = self._get_dynamic_chunk_size(response_size)

            # Receive response data
            response_data = b""
            while len(response_data) < response_size:
                remaining = response_size - len(response_data)
                chunk = client_socket.recv(min(chunk_size, remaining))
                if not chunk:
                    break
                response_data += chunk

            if len(response_data) != response_size:
                raise Exception(
                    f"Incomplete data: expected {response_size} bytes, got {len(response_data)}"
                )

            response = msgpack_decode(response_data)
            client_socket.close()

            if response["status"] == "error":
                raise Exception(response["message"])

            return response["data"]

        except Exception as e:
            print(f"DEBUG: Exception in _send_request: {e}")
            raise Exception(f"Cache server communication error: {e}")

    def get_response(
        self,
        prompt: str,
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Get single response from cache"""
        request = {
            "operation": "get_response",
            "prompt": prompt,
            "model_type": model_type,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        return self._send_request(request)

    def save_response(
        self,
        prompt: str,
        llm_output: str,
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
    ):
        """Save single response to cache"""
        request = {
            "operation": "save_response",
            "prompt": prompt,
            "llm_output": llm_output,
            "model_type": model_type,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        self._send_request(request)

    def get_responses(
        self,
        batch_prompts: List[Union[str, list]],
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
    ):
        """Get batch responses from cache"""
        request = {
            "operation": "get_responses",
            "batch_prompts": batch_prompts,
            "model_type": model_type,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        return self._send_request(request)

    def save_responses(
        self,
        batch_prompts: List[str],
        llm_outputs: List[str],
        model_type: LLMModel,
        seed: int,
        max_new_tokens: int,
        temperature: float,
    ):
        """Save batch responses to cache"""
        request = {
            "operation": "save_responses",
            "batch_prompts": batch_prompts,
            "llm_outputs": llm_outputs,
            "model_type": model_type,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }
        self._send_request(request)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cache_server.py <database_path> [port]")
        sys.exit(1)

    db_path = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    server = CacheServer(db_path, port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down cache server...")
        server.stop()
