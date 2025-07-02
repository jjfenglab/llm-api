"""
Tests for CacheServer and CacheClient functionality.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

from lab_llm.cache_server import (
    CacheClient,
    CacheServer,
    msgpack_decode,
    msgpack_encode,
)


class TestCacheServerBasicOperations:
    """Test basic cache server operations."""

    def test_cache_server_initialization(self, temp_db_path):
        """Test that cache server initializes properly."""
        server = CacheServer(temp_db_path, port=9990)

        assert server is not None
        assert server.db_path == temp_db_path
        assert server.port == 9990
        assert server.cache is not None
        assert server.handler is not None
        assert not server.running

    def test_cache_client_initialization_without_auto_start(self):
        """Test cache client initialization without auto-starting server."""
        client = CacheClient(port=9991)

        assert client is not None
        assert client.port == 9991
        assert not client._auto_started_server

    def test_cache_client_auto_start_with_cache_file(self, temp_db_path):
        """Test cache client auto-starting server when cache_file is provided."""
        # Use a different port to avoid conflicts
        port = 9992

        # Mock the server starting process to avoid actually starting a subprocess
        with patch.object(CacheClient, "_ensure_server_running") as mock_ensure:
            client = CacheClient(cache_file=temp_db_path, port=port)
            mock_ensure.assert_called_once()
            assert client.cache_file == temp_db_path
            assert client.port == port


class TestCacheServerRequestHandling:
    """Test cache server request handling."""

    def test_handle_get_response_request(self, temp_db_path, openai_model):
        """Test handling get_response requests."""
        server = CacheServer(temp_db_path, port=9993)

        # First save a response to have something to get
        save_request = {
            "operation": "save_response",
            "prompt": "test prompt",
            "llm_output": "test response",
            "model_type": openai_model,
            "seed": 42,
            "max_new_tokens": 100,
            "temperature": 0.0,
        }

        result = server.handle_request(msgpack_encode(save_request))
        assert result["status"] == "success"

        # Now test get_response
        get_request = {
            "operation": "get_response",
            "prompt": "test prompt",
            "model_type": openai_model,
            "seed": 42,
            "max_new_tokens": 100,
            "temperature": 0.0,
        }

        result = server.handle_request(msgpack_encode(get_request))
        assert result["status"] == "success"
        found, response = result["data"]
        assert found
        assert response == "test response"

    def test_handle_save_response_request(self, temp_db_path, openai_model):
        """Test handling save_response requests."""
        server = CacheServer(temp_db_path, port=9994)

        request = {
            "operation": "save_response",
            "prompt": "save test prompt",
            "llm_output": "save test response",
            "model_type": openai_model,
            "seed": 42,
            "max_new_tokens": 100,
            "temperature": 0.0,
        }

        result = server.handle_request(msgpack_encode(request))
        assert result["status"] == "success"
        assert result["data"] is None

    def test_handle_get_responses_batch_request(self, temp_db_path, openai_model):
        """Test handling batch get_responses requests."""
        server = CacheServer(temp_db_path, port=9995)

        # Save some responses first
        prompts = ["batch prompt 1", "batch prompt 2"]
        responses = ["batch response 1", "batch response 2"]

        save_request = {
            "operation": "save_responses",
            "batch_prompts": prompts,
            "llm_outputs": responses,
            "model_type": openai_model,
            "seed": 42,
            "max_new_tokens": 100,
            "temperature": 0.0,
        }

        result = server.handle_request(msgpack_encode(save_request))
        assert result["status"] == "success"

        # Now test get_responses
        get_request = {
            "operation": "get_responses",
            "batch_prompts": prompts,
            "model_type": openai_model,
            "seed": 42,
            "max_new_tokens": 100,
            "temperature": 0.0,
        }

        result = server.handle_request(msgpack_encode(get_request))
        assert result["status"] == "success"
        df = result["data"]
        assert len(df) == 2
        assert df["found_in_cache"].all()
        assert df["llm_output"].tolist() == responses

    def test_handle_unknown_operation(self, temp_db_path):
        """Test handling unknown operation requests."""
        server = CacheServer(temp_db_path, port=9996)

        request = {"operation": "unknown_operation", "some_param": "some_value"}

        result = server.handle_request(msgpack_encode(request))
        assert result["status"] == "error"
        assert "Unknown operation" in result["message"]

    def test_handle_malformed_request(self, temp_db_path):
        """Test handling malformed requests."""
        server = CacheServer(temp_db_path, port=9997)

        # Send invalid data
        result = server.handle_request(b"invalid_data")
        assert result["status"] == "error"
        assert "message" in result


class TestCacheServerSerialization:
    """Test serialization and deserialization."""

    def test_msgpack_encoding_decoding(self, openai_model):
        """Test msgpack encoding and decoding with LLMModel."""

        data = {
            "operation": "test",
            "model_type": openai_model,
            "prompt": "test prompt",
            "temperature": 0.5,
        }

        encoded = msgpack_encode(data)
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0

        decoded = msgpack_decode(encoded)
        assert decoded["operation"] == "test"
        assert decoded["model_type"].name.value == openai_model.name.value
        assert decoded["prompt"] == "test prompt"
        assert decoded["temperature"] == 0.5

    def test_pickle_fallback_for_large_data(self):
        """Test that large data falls back to pickle."""

        # Create large data that exceeds msgpack threshold
        large_data = {"operation": "test", "large_field": "x" * 300000}  # 300KB of data

        encoded = msgpack_encode(large_data)
        assert isinstance(encoded, bytes)
        assert encoded[0] == 1  # Should use pickle (magic byte 1)

        decoded = msgpack_decode(encoded)
        assert decoded["operation"] == "test"
        assert len(decoded["large_field"]) == 300000


class TestCacheServerHelperMethods:
    """Test cache server helper methods."""

    def test_get_optimal_chunk_size(self, temp_db_path):
        """Test optimal chunk size calculation."""
        server = CacheServer(temp_db_path, port=9998)

        chunk_size = server._get_optimal_chunk_size()
        assert isinstance(chunk_size, int)
        assert chunk_size > 0
        assert chunk_size == 64 * 1024  # 64KB default

    def test_get_dynamic_chunk_size(self, temp_db_path):
        """Test dynamic chunk size calculation based on data size."""
        server = CacheServer(temp_db_path, port=9999)

        # Small data
        small_chunk = server._get_dynamic_chunk_size(4096)  # 4KB
        assert small_chunk <= 4096

        # Medium data
        medium_chunk = server._get_dynamic_chunk_size(32 * 1024)  # 32KB
        assert medium_chunk <= 16 * 1024

        # Large data
        large_chunk = server._get_dynamic_chunk_size(128 * 1024)  # 128KB
        assert large_chunk == server.chunk_size


# Tests use msgpack_encode directly instead of a helper method


class TestCacheServerConcurrency:
    """Test cache server concurrent operations."""

    def test_concurrent_read_operations(self, temp_db_path, openai_model):
        """Test multiple clients reading from cache concurrently."""
        # Use a unique port to avoid conflicts
        port = 10001

        # Start a cache server in a separate thread
        server = CacheServer(temp_db_path, port=port)
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True

        try:
            server_thread.start()
            # Give server time to start
            time.sleep(0.5)

            # Pre-populate cache with test data
            client = CacheClient(port=port)
            test_data = [
                ("prompt_1", "response_1"),
                ("prompt_2", "response_2"),
                ("prompt_3", "response_3"),
            ]

            for prompt, response in test_data:
                client.save_response(
                    prompt=prompt,
                    llm_output=response,
                    model_type=openai_model,
                    seed=42,
                    max_new_tokens=100,
                    temperature=0.0,
                )

            # Define worker function for concurrent reads
            def read_worker(worker_id, num_operations):
                results = []
                test_client = CacheClient(port=port)

                for i in range(num_operations):
                    # Try to read different prompts
                    prompt_idx = (worker_id + i) % len(test_data)
                    prompt, expected_response = test_data[prompt_idx]

                    start_time = time.time()
                    found, response = test_client.get_response(
                        prompt=prompt,
                        model_type=openai_model,
                        seed=42,
                        max_new_tokens=100,
                        temperature=0.0,
                    )
                    duration = time.time() - start_time

                    results.append(
                        {
                            "worker_id": worker_id,
                            "operation": i,
                            "found": found,
                            "response": response,
                            "expected": expected_response,
                            "duration": duration,
                            "success": found and response == expected_response,
                        }
                    )

                return results

            # Run concurrent read operations
            num_workers = 4
            operations_per_worker = 5

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(read_worker, i, operations_per_worker)
                    for i in range(num_workers)
                ]

                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())

            # Analyze results
            successful_ops = [r for r in all_results if r["success"]]
            failed_ops = [r for r in all_results if not r["success"]]

            # Assertions
            assert len(all_results) == num_workers * operations_per_worker
            assert len(failed_ops) == 0, f"Failed operations: {failed_ops}"
            assert len(successful_ops) == len(all_results)

            # Check that all responses are correct
            for result in successful_ops:
                assert result["found"], f"Cache miss for operation: {result}"
                assert (
                    result["response"] == result["expected"]
                ), f"Wrong response: {result}"

            # Performance check - operations should be reasonably fast
            avg_duration = sum(r["duration"] for r in successful_ops) / len(
                successful_ops
            )
            assert (
                avg_duration < 0.1
            ), f"Operations too slow: {avg_duration:.3f}s average"

        finally:
            server.stop()
            # Give server time to stop
            time.sleep(0.1)

    def test_concurrent_write_and_read_operations(self, temp_db_path, openai_model):
        """Test mixed concurrent write and read operations."""
        # Use a unique port to avoid conflicts
        port = 10002

        # Start a cache server in a separate thread
        server = CacheServer(temp_db_path, port=port)
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True

        try:
            server_thread.start()
            # Give server time to start
            time.sleep(0.5)

            # Define worker functions
            def write_worker(worker_id, num_operations):
                results = []
                test_client = CacheClient(port=port)

                for i in range(num_operations):
                    prompt = f"write_prompt_{worker_id}_{i}"
                    response = f"write_response_{worker_id}_{i}"

                    start_time = time.time()
                    try:
                        test_client.save_response(
                            prompt=prompt,
                            llm_output=response,
                            model_type=openai_model,
                            seed=worker_id,
                            max_new_tokens=100,
                            temperature=0.0,
                        )
                        duration = time.time() - start_time
                        results.append(
                            {
                                "worker_id": worker_id,
                                "operation": i,
                                "type": "write",
                                "prompt": prompt,
                                "response": response,
                                "duration": duration,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "worker_id": worker_id,
                                "operation": i,
                                "type": "write",
                                "duration": time.time() - start_time,
                                "success": False,
                                "error": str(e),
                            }
                        )

                return results

            def read_worker(worker_id, num_operations):
                results = []
                test_client = CacheClient(port=port)

                for i in range(num_operations):
                    # Try to read from different writers
                    target_writer = worker_id % 2  # Read from writers 0 or 1
                    target_op = i % 3  # Read from operations 0, 1, or 2
                    prompt = f"write_prompt_{target_writer}_{target_op}"

                    start_time = time.time()
                    try:
                        found, response = test_client.get_response(
                            prompt=prompt,
                            model_type=openai_model,
                            seed=target_writer,
                            max_new_tokens=100,
                            temperature=0.0,
                        )
                        duration = time.time() - start_time
                        results.append(
                            {
                                "worker_id": worker_id,
                                "operation": i,
                                "type": "read",
                                "prompt": prompt,
                                "found": found,
                                "response": response,
                                "duration": duration,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        results.append(
                            {
                                "worker_id": worker_id,
                                "operation": i,
                                "type": "read",
                                "duration": time.time() - start_time,
                                "success": False,
                                "error": str(e),
                            }
                        )

                return results

            # Run mixed concurrent operations
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []

                # 2 writers
                for i in range(2):
                    futures.append(executor.submit(write_worker, i, 5))

                # Give writers a head start
                time.sleep(0.1)

                # 4 readers
                for i in range(4):
                    futures.append(executor.submit(read_worker, i + 10, 4))

                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())

            # Analyze results
            write_results = [r for r in all_results if r["type"] == "write"]
            read_results = [r for r in all_results if r["type"] == "read"]

            successful_writes = [r for r in write_results if r["success"]]
            successful_reads = [r for r in read_results if r["success"]]
            failed_operations = [r for r in all_results if not r["success"]]

            # Assertions
            assert (
                len(failed_operations) == 0
            ), f"Failed operations: {failed_operations}"
            assert (
                len(successful_writes) == 10
            ), f"Expected 10 writes, got {len(successful_writes)}"
            assert (
                len(successful_reads) == 16
            ), f"Expected 16 reads, got {len(successful_reads)}"

            # Check that some reads found cached data
            cache_hits = [r for r in successful_reads if r.get("found", False)]
            # We expect some cache hits since readers are looking for data written by writers
            assert (
                len(cache_hits) > 0
            ), "Expected some cache hits from concurrent operations"

        finally:
            server.stop()
            # Give server time to stop
            time.sleep(0.1)

    def test_batch_concurrent_operations(self, temp_db_path, openai_model):
        """Test concurrent batch operations."""
        # Use a unique port to avoid conflicts
        port = 10003

        # Start a cache server in a separate thread
        server = CacheServer(temp_db_path, port=port)
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True

        try:
            server_thread.start()
            # Give server time to start
            time.sleep(0.5)

            def batch_worker(worker_id, num_batches):
                results = []
                test_client = CacheClient(port=port)

                for batch_idx in range(num_batches):
                    # Create batch data
                    batch_prompts = [
                        f"batch_{worker_id}_{batch_idx}_prompt_{i}" for i in range(3)
                    ]
                    batch_responses = [
                        f"batch_{worker_id}_{batch_idx}_response_{i}" for i in range(3)
                    ]

                    start_time = time.time()
                    try:
                        # Save batch
                        test_client.save_responses(
                            batch_prompts=batch_prompts,
                            llm_outputs=batch_responses,
                            model_type=openai_model,
                            seed=worker_id,
                            max_new_tokens=100,
                            temperature=0.0,
                        )

                        # Read back batch
                        df = test_client.get_responses(
                            batch_prompts=batch_prompts,
                            model_type=openai_model,
                            seed=worker_id,
                            max_new_tokens=100,
                            temperature=0.0,
                        )

                        duration = time.time() - start_time

                        # Verify results
                        success = (
                            len(df) == 3
                            and df["found_in_cache"].all()
                            and df["llm_output"].tolist() == batch_responses
                        )

                        results.append(
                            {
                                "worker_id": worker_id,
                                "batch": batch_idx,
                                "duration": duration,
                                "success": success,
                                "prompts": batch_prompts,
                                "responses": batch_responses,
                            }
                        )

                    except Exception as e:
                        results.append(
                            {
                                "worker_id": worker_id,
                                "batch": batch_idx,
                                "duration": time.time() - start_time,
                                "success": False,
                                "error": str(e),
                            }
                        )

                return results

            # Run concurrent batch operations
            num_workers = 3
            batches_per_worker = 3

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(batch_worker, i, batches_per_worker)
                    for i in range(num_workers)
                ]

                all_results = []
                for future in as_completed(futures):
                    all_results.extend(future.result())

            # Analyze results
            successful_ops = [r for r in all_results if r["success"]]
            failed_ops = [r for r in all_results if not r["success"]]

            # Assertions
            expected_total = num_workers * batches_per_worker
            assert len(all_results) == expected_total
            assert len(failed_ops) == 0, f"Failed operations: {failed_ops}"
            assert len(successful_ops) == expected_total

        finally:
            server.stop()
            # Give server time to stop
            time.sleep(0.1)


# All fixtures are now imported from conftest.py
