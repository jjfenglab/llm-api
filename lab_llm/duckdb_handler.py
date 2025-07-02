import random
import time

import duckdb


class DuckDBHandler:
    def __init__(self, db_path: str, read_only: bool = False, max_retries: int = 5):
        """Initialize the DuckDB handler.

        Args:
            db_path (str): Path to the DuckDB database file
            read_only (bool, optional): Whether to open the connection in read-only mode. Defaults to False.
            max_retries (int, optional): Maximum number of retry attempts for database operations. Defaults to 5.
        """
        self.db_path = db_path
        self.read_only = read_only
        self.max_retries = max_retries
        self.conn = None
        self._connect_with_retry()

    def close_connection(self):
        if self.conn:
            self.conn.close()

    def get_connection(self):
        return self.conn

    def _connect_with_retry(self):
        """Establish database connection with retry logic for handling concurrent access."""
        for attempt in range(self.max_retries):
            try:
                self.conn = duckdb.connect(self.db_path, read_only=self.read_only)
                return
            except Exception as e:
                error_str = str(e).lower()
                if "conflicting lock" in error_str or "database is locked" in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = random.uniform(0.1, 0.5)
                        print(
                            f"Database connection conflict detected. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                raise Exception(
                    f"Failed to connect to database after {self.max_retries} attempts: {e}"
                )

    def execute_with_retry(self, query, params=None):
        """Execute a query with retry logic for concurrent access."""
        for attempt in range(self.max_retries):
            try:
                if params:
                    return self.conn.execute(query, params)
                else:
                    return self.conn.execute(query)
            except Exception as e:
                error_str = str(e).lower()
                if "conflicting lock" in error_str or "database is locked" in error_str:
                    if attempt < self.max_retries - 1:
                        wait_time = random.uniform(0.1, 0.5)
                        print(
                            f"Database operation conflict detected. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                raise Exception(
                    f"Failed to execute query after {self.max_retries} attempts: {e}"
                )
