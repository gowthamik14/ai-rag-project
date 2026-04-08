"""
Singleton Firestore client.

Reads credentials from the path specified in Settings and creates a single
google.cloud.firestore.Client that every other module reuses.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import google.auth
from google.cloud import firestore
from google.oauth2 import service_account

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_client: Optional[firestore.Client] = None


def get_firestore_client() -> firestore.Client:
    """Return the module-level singleton Firestore client (thread-safe)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_client()
    return _client


def _build_client() -> firestore.Client:
    cred_path = settings.google_application_credentials
    project = settings.firestore_project_id

    if os.path.exists(cred_path):
        credentials = service_account.Credentials.from_service_account_file(cred_path)
        client = firestore.Client(project=project, credentials=credentials)
        logger.info("Firestore client initialised with service-account file: %s", cred_path)
    else:
        # Fall back to Application Default Credentials (e.g. Cloud Run, GKE)
        credentials, detected_project = google.auth.default()
        client = firestore.Client(
            project=project or detected_project,
            credentials=credentials,
        )
        logger.info("Firestore client initialised with Application Default Credentials.")

    return client


class FirestoreClient:
    """
    Thin wrapper around google.cloud.firestore that provides:
      - CRUD helpers with typed returns
      - Batch write support
      - Query helpers used by DocumentStore
    """

    def __init__(self) -> None:
        self._db = get_firestore_client()

    # ------------------------------------------------------------------ #
    # Generic CRUD
    # ------------------------------------------------------------------ #

    def set(self, collection: str, doc_id: str, data: Dict[str, Any]) -> str:
        """Create or fully overwrite a document. Returns doc_id."""
        self._db.collection(collection).document(doc_id).set(data)
        logger.debug("SET %s/%s", collection, doc_id)
        return doc_id

    def merge(self, collection: str, doc_id: str, data: Dict[str, Any]) -> str:
        """Create or partially update a document (merge=True). Returns doc_id."""
        self._db.collection(collection).document(doc_id).set(data, merge=True)
        logger.debug("MERGE %s/%s", collection, doc_id)
        return doc_id

    def get(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single document. Returns None if missing."""
        snap = self._db.collection(collection).document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    def delete(self, collection: str, doc_id: str) -> None:
        self._db.collection(collection).document(doc_id).delete()
        logger.debug("DELETE %s/%s", collection, doc_id)

    def list_all(self, collection: str) -> List[Dict[str, Any]]:
        """Return all documents in a collection (use carefully on large sets)."""
        docs = self._db.collection(collection).stream()
        return [{"id": d.id, **d.to_dict()} for d in docs]

    def query(
        self,
        collection: str,
        filters: Optional[List[tuple]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Flexible query helper.

        filters: list of (field, op, value) tuples, e.g. [("status", "==", "active")]
        order_by: field name to sort by
        limit: max documents to return
        """
        ref = self._db.collection(collection)

        if filters:
            for field, op, value in filters:
                ref = ref.where(field, op, value)

        if order_by:
            ref = ref.order_by(order_by)

        if limit:
            ref = ref.limit(limit)

        return [{"id": d.id, **d.to_dict()} for d in ref.stream()]

    # ------------------------------------------------------------------ #
    # Batch writes
    # ------------------------------------------------------------------ #

    def batch_set(self, collection: str, items: List[Dict[str, Any]], id_field: str = "id") -> int:
        """
        Write many documents in Firestore batches (max 500 per batch).
        Each item must contain the field named by `id_field`.
        Returns number of documents written.
        """
        BATCH_LIMIT = 500
        written = 0
        for i in range(0, len(items), BATCH_LIMIT):
            batch = self._db.batch()
            chunk = items[i : i + BATCH_LIMIT]
            for item in chunk:
                doc_id = str(item[id_field])
                ref = self._db.collection(collection).document(doc_id)
                batch.set(ref, item)
            batch.commit()
            written += len(chunk)
            logger.debug("Batch committed %d docs to %s", len(chunk), collection)

        return written
