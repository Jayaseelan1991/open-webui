from typing import Optional, List, Dict, Any, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor, Json, execute_batch
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.pool import ThreadedConnectionPool

from open_webui.apps.retrieval.vector.main import VectorItem, SearchResult, GetResult
from open_webui.config import PGVECTOR_URI, PGVECTOR_CONNECTION_POOL_SIZE

import json

NO_LIMIT: int = 999999999

# Register an adapter to handle the vector type
class Vector:
    def __init__(self, data: List[float]):
        self.data: List[float] = data

def vector_adapter(vector: 'Vector') -> AsIs:
    vector_str = '[' + ','.join(map(str, vector.data)) + ']'
    return AsIs("'" + vector_str + "'::vector")

register_adapter(Vector, vector_adapter)

class PgvectorClient:
    def __init__(self) -> None:
        self.collection_prefix: str = "open_webui"
        self.PGVECTOR_URI: str = PGVECTOR_URI
        self.PGVECTOR_CONNECTION_POOL_SIZE: int = PGVECTOR_CONNECTION_POOL_SIZE
        if self.PGVECTOR_URI:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=self.PGVECTOR_CONNECTION_POOL_SIZE,
                dsn=self.PGVECTOR_URI
            )
            # Initialize the extension in a separate connection
            conn = self.pool.getconn()
            try:
                self.enable_pgvector_extension(conn)
            finally:
                self.pool.putconn(conn)
        else:
            self.pool = None

    def enable_pgvector_extension(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    def _get_connection(self):
        if not self.pool:
            raise Exception("Connection pool is not initialized.")
        return self.pool.getconn()

    def _put_connection(self, conn):
        if self.pool:
            self.pool.putconn(conn)

    def _result_to_get_result(self, results: List[Dict[str, Any]]) -> GetResult:
        ids: List[List[str]] = []
        documents: List[List[str]] = []
        metadatas: List[List[Any]] = []

        _ids: List[str] = []
        _documents: List[str] = []
        _metadatas: List[Any] = []

        for row in results:
            _ids.append(row['id'])
            _documents.append(row['text'])
            _metadatas.append(row['metadata'])

        ids.append(_ids)
        documents.append(_documents)
        metadatas.append(_metadatas)

        return GetResult(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def _result_to_search_result(self, results: List[Dict[str, Any]]) -> SearchResult:
        ids: List[List[str]] = []
        distances: List[List[float]] = []
        documents: List[List[str]] = []
        metadatas: List[List[Any]] = []

        _ids: List[str] = []
        _distances: List[float] = []
        _documents: List[str] = []
        _metadatas: List[Any] = []

        for row in results:
            _ids.append(row['id'])
            _distances.append(row['distance'])
            _documents.append(row['text'])
            _metadatas.append(row['metadata'])

        ids.append(_ids)
        distances.append(_distances)
        documents.append(_documents)
        metadatas.append(_metadatas)

        return SearchResult(
            ids=ids,
            distances=distances,
            documents=documents,
            metadatas=metadatas,
        )

    def _create_collection(self, collection_name: str, dimension: int) -> None:
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor() as cur:
                create_table_sql: str = f"""
                CREATE TABLE IF NOT EXISTS {collection_name_with_prefix} (
                    id TEXT PRIMARY KEY,
                    vector VECTOR({dimension}),
                    text TEXT,
                    metadata JSONB
                );
                """
                cur.execute(create_table_sql)
                create_index_sql: str = f"""
                CREATE INDEX IF NOT EXISTS {collection_name_with_prefix}_vector_idx
                ON {collection_name_with_prefix} USING ivfflat (vector) WITH (lists = 100);
                """
                cur.execute(create_index_sql)
            conn.commit()
            print(f"Collection {collection_name_with_prefix} successfully created!")
        finally:
            self._put_connection(conn)

    def _create_collection_if_not_exists(self, collection_name: str, dimension: int) -> None:
        if not self.has_collection(collection_name):
            self._create_collection(collection_name, dimension)

    def has_collection(self, collection_name: str) -> bool:
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    );
                    """,
                    (collection_name_with_prefix,),
                )
                exists: bool = cur.fetchone()[0]
            return exists
        finally:
            self._put_connection(conn)

    def delete_collection(self, collection_name: str) -> None:
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor() as cur:
                delete_table_sql: str = f"DROP TABLE IF EXISTS {collection_name_with_prefix} CASCADE;"
                cur.execute(delete_table_sql)
            conn.commit()
        finally:
            self._put_connection(conn)

    def search(
        self, collection_name: str, vectors: List[List[float]], limit: Optional[int]
    ) -> Optional[SearchResult]:
        if limit is None:
            limit = NO_LIMIT
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            query_vector: List[float] = vectors[0]
            query_vector_obj: Vector = Vector(query_vector)

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                search_sql: str = f"""
                SELECT id, text, metadata, (vector <=> %s) AS distance
                FROM {collection_name_with_prefix}
                ORDER BY vector <=> %s
                LIMIT %s;
                """
                cur.execute(search_sql, (query_vector_obj, query_vector_obj, limit))
                results: List[Dict[str, Any]] = cur.fetchall()

            if not results:
                return None

            return self._result_to_search_result(results)
        finally:
            self._put_connection(conn)

    def query(
        self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None
    ) -> Optional[GetResult]:
        if not self.has_collection(collection_name):
            return None
        if limit is None:
            limit = NO_LIMIT

        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                where_clauses: List[str] = []
                values: List[Any] = []
                for key, value in filter.items():
                    where_clauses.append("metadata ->> %s = %s")
                    values.extend([key, json.dumps(value)])
                where_sql: str = ' AND '.join(where_clauses)

                query_sql: str = f"""
                SELECT id, text, metadata
                FROM {collection_name_with_prefix}
                WHERE {where_sql}
                LIMIT %s;
                """
                values.append(limit)
                cur.execute(query_sql, values)
                results: List[Dict[str, Any]] = cur.fetchall()

            if not results:
                return None

            return self._result_to_get_result(results)
        except Exception as e:
            print(f"Error when querying the collection: {e}")
            return None
        finally:
            self._put_connection(conn)

    def get(self, collection_name: str) -> Optional[GetResult]:
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                get_sql: str = f"""
                SELECT id, text, metadata
                FROM {collection_name_with_prefix}
                LIMIT %s;
                """
                cur.execute(get_sql, (NO_LIMIT,))
                results: List[Dict[str, Any]] = cur.fetchall()

            if not results:
                return None

            return self._result_to_get_result(results)
        finally:
            self._put_connection(conn)

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")

            with conn.cursor() as cur:
                insert_sql: str = f"""
                INSERT INTO {collection_name_with_prefix} (id, vector, text, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
                """
                data: List[Tuple[str, Vector, str, Json]] = [
                    (
                        item["id"],
                        Vector(item["vector"]),
                        item["text"],
                        Json(item["metadata"]),
                    )
                    for item in items
                ]
                execute_batch(cur, insert_sql, data)
            conn.commit()
        finally:
            self._put_connection(conn)

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        self._create_collection_if_not_exists(collection_name, len(items[0]["vector"]))
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor() as cur:
                upsert_sql: str = f"""
                INSERT INTO {collection_name_with_prefix} (id, vector, text, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                vector = EXCLUDED.vector,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata;
                """
                data: List[Tuple[str, Vector, str, Json]] = [
                    (
                        item["id"],
                        Vector(item["vector"]),
                        item["text"],
                        Json(item["metadata"]),
                    )
                    for item in items
                ]
                execute_batch(cur, upsert_sql, data)
            conn.commit()
        finally:
            self._put_connection(conn)

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        conn = self._get_connection()
        try:
            collection_name_with_prefix: str = f"{self.collection_prefix}_{collection_name}"
            collection_name_with_prefix = collection_name_with_prefix.replace("-", "_")
            with conn.cursor() as cur:
                if ids:
                    delete_sql: str = f"""
                    DELETE FROM {collection_name_with_prefix}
                    WHERE id IN %s;
                    """
                    cur.execute(delete_sql, (tuple(ids),))
                elif filter:
                    where_clauses: List[str] = []
                    values: List[Any] = []
                    for key, value in filter.items():
                        where_clauses.append("metadata ->> %s = %s")
                        values.extend([key, json.dumps(value)])
                    where_sql: str = ' AND '.join(where_clauses)
                    delete_sql: str = f"""
                    DELETE FROM {collection_name_with_prefix}
                    WHERE {where_sql};
                    """
                    cur.execute(delete_sql, values)
                else:
                    delete_sql: str = f"DELETE FROM {collection_name_with_prefix};"
                    cur.execute(delete_sql)
            conn.commit()
        finally:
            self._put_connection(conn)

    def reset(self) -> None:
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name LIKE %s;
                    """,
                    (f"{self.collection_prefix}_%",),
                )
                tables: List[Dict[str, Any]] = cur.fetchall()
                for table in tables:
                    table_name: str = table['table_name']
                    cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            conn.commit()
        finally:
            self._put_connection(conn)

    def close(self) -> None:
        if self.pool:
            self.pool.closeall()