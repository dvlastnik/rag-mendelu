import re
import pathlib
from typing import List

import duckdb
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)

_MUTATION_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b',
    re.IGNORECASE
)


class SqlDbRepository:
    """DuckDB-backed repository for CSV/XLSX analytical queries.

    Used alongside Qdrant: Qdrant handles semantic similarity search,
    DuckDB handles aggregation/ranking queries (AVG, COUNT, ORDER BY, etc.)
    on tabular data. Works with any column schema — no pre-configuration needed.
    """

    def __init__(self, db_path: str = 'data/sql/tabular.duckdb') -> None:
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        logger.info(f"SqlDbRepository connected to {db_path}")

    def register_csv(self, source: str, file_path: str) -> None:
        """Register a CSV file as a DuckDB table. Idempotent (CREATE OR REPLACE)."""
        table = self._safe_table_name(source)
        try:
            self.conn.execute(
                f"CREATE OR REPLACE TABLE \"{table}\" AS SELECT * FROM read_csv_auto('{file_path}')"
            )
            count = self.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            logger.info(f"Registered CSV '{file_path}' as table '{table}' ({count} rows)")
        except Exception as e:
            logger.error(f"Failed to register CSV '{file_path}' as '{table}': {e}")

    def register_xlsx(self, source: str, file_path: str) -> None:
        """Register an XLSX file as a DuckDB table via pandas. Idempotent."""
        table = self._safe_table_name(source)
        try:
            df = pd.read_excel(file_path)
            self.conn.register('_temp_xlsx', df)
            self.conn.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM _temp_xlsx')
            self.conn.unregister('_temp_xlsx')
            count = self.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            logger.info(f"Registered XLSX '{file_path}' as table '{table}' ({count} rows)")
        except Exception as e:
            logger.error(f"Failed to register XLSX '{file_path}' as '{table}': {e}")

    def get_schema(self, table_name: str) -> str:
        """Return DESCRIBE output + 3 sample rows as a formatted string for LLM context."""
        table = self._safe_table_name(table_name)
        try:
            describe_df = self.conn.execute(f'DESCRIBE "{table}"').df()
            sample_df = self.conn.execute(f'SELECT * FROM "{table}" LIMIT 3').df()
            schema_str = f"Table: {table}\n\nColumns:\n{describe_df.to_string(index=False)}\n\nSample rows:\n{sample_df.to_string(index=False)}"
            return schema_str
        except Exception as e:
            logger.error(f"Failed to get schema for '{table}': {e}")
            return f"Table: {table}\n(schema unavailable: {e})"

    def run_select(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT query and return a DataFrame. Raises on mutation SQL."""
        if _MUTATION_PATTERN.search(sql):
            raise ValueError(f"Mutation SQL is not allowed: {sql[:100]}")
        return self.conn.execute(sql).df()

    def list_tables(self) -> List[str]:
        """Return all registered table names."""
        try:
            rows = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in rows]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    def drop_table(self, source: str) -> None:
        """Drop a table (called during --erase ETL to clean up)."""
        table = self._safe_table_name(source)
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            logger.info(f"Dropped table '{table}'")
        except Exception as e:
            logger.warning(f"Failed to drop table '{table}': {e}")

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    @staticmethod
    def _safe_table_name(name: str) -> str:
        """Sanitize a source name to a safe DuckDB table name."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)
