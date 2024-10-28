from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    Text,
    text,
)
from sqlalchemy import text, bindparam
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import Integer
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.declarative import declarative_base

from open_webui.apps.webui.internal.db import Session
from open_webui.apps.retrieval.vector.main import VectorItem, SearchResult, GetResult

VECTOR_LENGTH = 1536
Base = declarative_base()


class DocumentChunk(Base):
    __tablename__ = "document_chunk"

    id = Column(Text, primary_key=True)
    vector = Column(Vector(dim=VECTOR_LENGTH), nullable=True)
    collection_name = Column(Text, nullable=False)
    text = Column(Text, nullable=True)
    vmetadata = Column(MutableDict.as_mutable(JSONB), nullable=True)


class PgvectorClient:
    def __init__(self) -> None:
        self.session = Session
        try:
            # Ensure the pgvector extension is available
            self.session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            # Create the tables if they do not exist
            # Base.metadata.create_all requires a bind (engine or connection)
            # Get the connection from the session
            connection = self.session.connection()
            Base.metadata.create_all(bind=connection)

            # Create an index on the vector column if it doesn't exist
            self.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunk_vector "
                    "ON document_chunk USING ivfflat (vector) WITH (lists = 100);"
                )
            )
            self.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_document_chunk_collection_name "
                    "ON document_chunk (collection_name);"
                )
            )
            self.session.commit()
            print("Initialization complete.")
        except Exception as e:
            self.session.rollback()
            print(f"Error during initialization: {e}")
            raise

    def adjust_vector_length(self, vector: List[float]) -> List[float]:
        # Adjust vector to have length VECTOR_LENGTH
        current_length = len(vector)
        if current_length < VECTOR_LENGTH:
            # Pad the vector with zeros
            vector += [0.0] * (VECTOR_LENGTH - current_length)
        elif current_length > VECTOR_LENGTH:
            raise Exception(
                f"Vector length {current_length} not supported. Max length must be <= {VECTOR_LENGTH}"
            )
        return vector

    def insert(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            new_items = []
            for item in items:
                vector = self.adjust_vector_length(item["vector"])
                new_chunk = DocumentChunk(
                    id=item["id"],
                    vector=vector,
                    collection_name=collection_name,
                    text=item["text"],
                    vmetadata=item["metadata"],
                )
                new_items.append(new_chunk)
            self.session.bulk_save_objects(new_items)
            self.session.commit()
            print(
                f"Inserted {len(new_items)} items into collection '{collection_name}'."
            )
        except Exception as e:
            self.session.rollback()
            print(f"Error during insert: {e}")
            raise

    def upsert(self, collection_name: str, items: List[VectorItem]) -> None:
        try:
            for item in items:
                vector = self.adjust_vector_length(item["vector"])
                existing = (
                    self.session.query(DocumentChunk)
                    .filter(DocumentChunk.id == item["id"])
                    .first()
                )
                if existing:
                    existing.vector = vector
                    existing.text = item["text"]
                    existing.vmetadata = item["metadata"]
                    existing.collection_name = (
                        collection_name  # Update collection_name if necessary
                    )
                else:
                    new_chunk = DocumentChunk(
                        id=item["id"],
                        vector=vector,
                        collection_name=collection_name,
                        text=item["text"],
                        vmetadata=item["metadata"],
                    )
                    self.session.add(new_chunk)
            self.session.commit()
            print(f"Upserted {len(items)} items into collection '{collection_name}'.")
        except Exception as e:
            self.session.rollback()
            print(f"Error during upsert: {e}")
            raise

    def search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        limit: Optional[int] = None,
    ) -> Optional[SearchResult]:
        try:
            if not vectors:
                return None

            # Adjust query vectors to VECTOR_LENGTH
            vectors = [self.adjust_vector_length(vector) for vector in vectors]
            num_queries = len(vectors)

            # Build the VALUES clause for the query_vectors CTE
            # And collect the parameters to bind
            values_clause = []
            params = {"collection_name": collection_name}
            if limit is not None:
                params["limit"] = limit

            for idx, vector in enumerate(vectors):
                # Use :q_vector_n as parameter placeholders
                param_name = f"q_vector_{idx}"
                params[param_name] = vector
                values_clause.append(f"({idx}, VECTOR(:{param_name}))")

            values_clause_str = ",\n".join(values_clause)

            # Build the SQL statement
            sql_query = text(
                f"""
                WITH query_vectors (qid, q_vector) AS (
                    VALUES
                    {values_clause_str}
                )
                SELECT
                    q.qid,
                    result.id,
                    result.text,
                    result.vmetadata,
                    result.distance
                FROM
                    query_vectors q
                JOIN LATERAL (
                    SELECT
                        d.id,
                        d.text,
                        d.vmetadata,
                        (d.vector <=> q.q_vector) AS distance
                    FROM
                        document_chunk d
                    WHERE
                        d.collection_name = :collection_name
                    ORDER BY
                        d.vector <=> q.q_vector
                    {'' if limit is None else 'LIMIT :limit'}
                ) result ON TRUE
                ORDER BY
                    q.qid, result.distance
            """
            )
            print("--------------------------")
            print(values_clause)
            print("--------------------------")
            print(sql_query)
            print("--------------------------")

            # Specify parameter types
            bind_params = [
                bindparam(f"q_vector_{idx}", type_=Vector(VECTOR_LENGTH))
                for idx in range(num_queries)
            ]
            if limit is not None:
                bind_params.append(bindparam("limit", type_=Integer))

            sql_query = sql_query.bindparams(*bind_params)

            # Execute the query
            result_proxy = self.session.execute(sql_query, params)
            results = result_proxy.mappings().all()

            # Organize results per query vector
            ids = [[] for _ in range(num_queries)]
            distances = [[] for _ in range(num_queries)]
            documents = [[] for _ in range(num_queries)]
            metadatas = [[] for _ in range(num_queries)]

            if not results:
                # Since we have multiple query vectors, we should return empty lists for each vector
                return SearchResult(
                    ids=ids,
                    distances=distances,
                    documents=documents,
                    metadatas=metadatas,
                )

            for row in results:
                qid = int(row["qid"])
                ids[qid].append(row["id"])
                distances[qid].append(row["distance"])
                documents[qid].append(row["text"])
                metadatas[qid].append(row["vmetadata"])

            return SearchResult(
                ids=ids, distances=distances, documents=documents, metadatas=metadatas
            )
        except Exception as e:
            print(f"Error during search: {e}")
            return None

    def query(
        self, collection_name: str, filter: Dict[str, Any], limit: Optional[int] = None
    ) -> Optional[GetResult]:
        try:
            query = self.session.query(DocumentChunk).filter(
                DocumentChunk.collection_name == collection_name
            )

            for key, value in filter.items():
                query = query.filter(DocumentChunk.vmetadata[key].astext == str(value))

            if limit is not None:
                query = query.limit(limit)

            results = query.all()

            if not results:
                return None

            ids = [[result.id for result in results]]
            documents = [[result.text for result in results]]
            metadatas = [[result.vmetadata for result in results]]

            return GetResult(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            print(f"Error during query: {e}")
            return None

    def get(
        self, collection_name: str, limit: Optional[int] = None
    ) -> Optional[GetResult]:
        try:
            query = self.session.query(DocumentChunk).filter(
                DocumentChunk.collection_name == collection_name
            )
            if limit is not None:
                query = query.limit(limit)

            results = query.all()

            if not results:
                return None

            ids = [[result.id for result in results]]
            documents = [[result.text for result in results]]
            metadatas = [[result.vmetadata for result in results]]

            return GetResult(ids=ids, documents=documents, metadatas=metadatas)
        except Exception as e:
            print(f"Error during get: {e}")
            return None

    def delete(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            query = self.session.query(DocumentChunk).filter(
                DocumentChunk.collection_name == collection_name
            )
            if ids:
                query = query.filter(DocumentChunk.id.in_(ids))
            if filter:
                for key, value in filter.items():
                    query = query.filter(
                        DocumentChunk.vmetadata[key].astext == str(value)
                    )
            deleted = query.delete(synchronize_session=False)
            self.session.commit()
            print(f"Deleted {deleted} items from collection '{collection_name}'.")
        except Exception as e:
            self.session.rollback()
            print(f"Error during delete: {e}")
            raise

    def reset(self) -> None:
        try:
            deleted = self.session.query(DocumentChunk).delete()
            self.session.commit()
            print(
                f"Reset complete. Deleted {deleted} items from 'document_chunk' table."
            )
        except Exception as e:
            self.session.rollback()
            print(f"Error during reset: {e}")
            raise

    def close(self) -> None:
        pass

    def has_collection(self, collection_name: str) -> bool:
        try:
            exists = (
                self.session.query(DocumentChunk)
                .filter(DocumentChunk.collection_name == collection_name)
                .first()
                is not None
            )
            return exists
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False

    def delete_collection(self, collection_name: str) -> None:
        self.delete(collection_name)
        print(f"Collection '{collection_name}' deleted.")
