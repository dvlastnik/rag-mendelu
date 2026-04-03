import re
from typing import List

import pandas as pd
from sqlalchemy import create_engine, text

from utils.logging_config import get_logger

logger = get_logger(__name__)

_MUTATION_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE)\b',
    re.IGNORECASE
)

_TYPE_NORMALIZATION = {
    'character varying': 'varchar',
    'text': 'varchar',
    'double precision': 'double',
    'timestamp without time zone': 'timestamp',
    'timestamp with time zone': 'timestamptz',
}


class PostgresqlRepository:
    """PostgreSQL-backed repository for CSV/XLSX analytical queries.

    Used alongside Qdrant: Qdrant handles semantic similarity search,
    PostgreSQL handles aggregation/ranking queries (AVG, COUNT, ORDER BY, etc.)
    on tabular data. Works with any column schema — no pre-configuration needed.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 5432,
        db: str = 'rag_mendelu',
        user: str = 'rag',
        password: str = 'rag_password',
    ) -> None:
        url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"
        self.engine = create_engine(url)
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"PostgresqlRepository connected to {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"PostgresqlRepository connection failed: {e}")
            raise

    def register_csv(self, source: str, file_path: str) -> None:
        """Register a CSV file as a PostgreSQL table. Idempotent (replaces existing)."""
        table = self._safe_table_name(source)
        try:
            df = pd.read_csv(file_path)
            df.to_sql(table, self.engine, if_exists='replace', index=False)
            count = self._count_rows(table)
            logger.info(f"Registered CSV '{file_path}' as table '{table}' ({count} rows)")
        except Exception as e:
            logger.error(f"Failed to register CSV '{file_path}' as '{table}': {e}")

    def register_xlsx(self, source: str, file_path: str) -> None:
        """Register an XLSX file as a PostgreSQL table. Idempotent."""
        table = self._safe_table_name(source)
        try:
            df = pd.read_excel(file_path)
            df.to_sql(table, self.engine, if_exists='replace', index=False)
            count = self._count_rows(table)
            logger.info(f"Registered XLSX '{file_path}' as table '{table}' ({count} rows)")
        except Exception as e:
            logger.error(f"Failed to register XLSX '{file_path}' as '{table}': {e}")

    def register_dataframe(self, source: str, df: pd.DataFrame) -> None:
        """Register an in-memory DataFrame as a PostgreSQL table. Idempotent."""
        table = self._safe_table_name(source)
        try:
            df.to_sql(table, self.engine, if_exists='replace', index=False)
            count = self._count_rows(table)
            logger.info(f"Registered DataFrame as table '{table}' ({count} rows)")
        except Exception as e:
            logger.error(f"Failed to register DataFrame as '{table}': {e}")

    def get_schema(self, table_name: str) -> str:
        """Return column info + 3 sample rows as a formatted string for LLM context."""
        table = self._safe_table_name(table_name)
        try:
            with self.engine.connect() as conn:
                cols_df = pd.read_sql(text(
                    "SELECT column_name, data_type AS column_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :t "
                    "ORDER BY ordinal_position"
                ), conn, params={"t": table})
                cols_df['column_type'] = cols_df['column_type'].map(
                    lambda t: _TYPE_NORMALIZATION.get(t, t)
                )
                sample_df = pd.read_sql(
                    text(f'SELECT * FROM "{table}" LIMIT 3'), conn
                )
            schema_str = (
                f"Table: {table}\n\n"
                f"Columns:\n{cols_df.to_string(index=False)}\n\n"
                f"Sample rows:\n{sample_df.to_string(index=False)}"
            )
            return schema_str
        except Exception as e:
            logger.error(f"Failed to get schema for '{table}': {e}")
            return f"Table: {table}\n(schema unavailable: {e})"

    def get_compact_catalog(self, available_sources: list[str] | None = None) -> str:
        """One-line-per-table compact schema summary for LLM context injection.

        Format: table_name(col1:type, col2:type, ...) [N rows]
        Returns empty string if no tables are registered.
        When available_sources is provided, only tables whose name is in that list are included
        (used to scope the catalog to the active Qdrant collection).
        """
        lines = []
        all_tables = self.list_tables()
        safe_sources = (
            {PostgresqlRepository._safe_table_name(s) for s in available_sources}
            if available_sources is not None else None
        )
        filtered = (
            [
                t for t in all_tables
                if t in safe_sources
                or any(t.startswith(s + '_table') for s in safe_sources)
            ]
            if safe_sources is not None
            else all_tables
        )
        for table in filtered:
            try:
                with self.engine.connect() as conn:
                    cols = pd.read_sql(text(
                        "SELECT column_name, data_type "
                        "FROM information_schema.columns "
                        "WHERE table_schema = 'public' AND table_name = :t "
                        "ORDER BY ordinal_position"
                    ), conn, params={"t": table})
                    col_strs = [
                        f"{row['column_name']}:{_TYPE_NORMALIZATION.get(row['data_type'], row['data_type'])}"
                        for _, row in cols.iterrows()
                    ]
                    count = self._count_rows(table)
                    lines.append(f"{table}({', '.join(col_strs)}) [{count} rows]")
            except Exception as e:
                logger.warning(f"get_compact_catalog: skipping '{table}': {e}")
        return "\n".join(lines)

    def get_table_names_catalog(self, available_sources: list[str] | None = None) -> str:
        """Return table names + row counts only (no column details) for QueryPlanner context.

        Keeps the QueryPlanner prompt small so the model only decides *which* tables to query;
        full column details are fetched later in analytical_query_agent via get_schema().
        """
        all_tables = self.list_tables()
        safe_sources = (
            {PostgresqlRepository._safe_table_name(s) for s in available_sources}
            if available_sources is not None else None
        )
        filtered = (
            [
                t for t in all_tables
                if t in safe_sources
                or any(t.startswith(s + '_table') for s in safe_sources)
            ]
            if safe_sources is not None
            else all_tables
        )
        lines = []
        for table in filtered:
            try:
                count = self._count_rows(table)
                lines.append(f"{table} [{count} rows]")
            except Exception:
                lines.append(table)
        return "\n".join(lines)

    def run_select(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT query and return a DataFrame. Raises on mutation SQL."""
        if _MUTATION_PATTERN.search(sql):
            raise ValueError(f"Mutation SQL is not allowed: {sql[:100]}")
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn)

    def list_tables(self) -> List[str]:
        """Return all registered table names."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                ))
                return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            return []

    def drop_table(self, source: str) -> None:
        """Drop a table (called during --erase ETL to clean up)."""
        table = self._safe_table_name(source)
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table}" CASCADE'))
                conn.commit()
            logger.info(f"Dropped table '{table}'")
        except Exception as e:
            logger.warning(f"Failed to drop table '{table}': {e}")

    def close(self) -> None:
        try:
            self.engine.dispose()
        except Exception:
            pass

    def _count_rows(self, table: str) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
            return result.scalar()

    @staticmethod
    def _safe_table_name(name: str) -> str:
        """Sanitize a source name to a safe PostgreSQL table name."""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()
