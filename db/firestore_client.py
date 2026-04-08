"""
Singleton Firestore client with graceful degradation.

If Firestore is disabled (FIRESTORE_ENABLED=false) or credentials are
placeholder/missing, all operations become silent no-ops and the app
continues to run in local-only mode (FAISS + in-memory sessions).
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_client = None          # google.cloud.firestore.Client or None
_client_initialised = False


def get_firestore_client():
    """Return the Firestore client, or None if unavailable."""
    global _client, _client_initialised
    if not _client_initialised:
        with _lock:
            if not _client_initialised:
                _client = _try_build_client()
                _client_initialised = True
    return _client


def _try_build_client():
    if not settings.firestore_enabled:
        logger.warning(
            "Firestore is DISABLED (FIRESTORE_ENABLED=false). "
            "Running in local-only mode — sessions and document persistence unavailable."
        )
        return None

    cred_path = settings.google_application_credentials
    project   = settings.firestore_project_id

    try:
        if os.path.exists(cred_path):
            from google.oauth2 import service_account
            from google.cloud import firestore
            credentials = service_account.Credentials.from_service_account_file(cred_path)
            client = firestore.Client(project=project, credentials=credentials)
            logger.info("Firestore connected via service-account: %s", cred_path)
            return client
        else:
            import google.auth
            from google.cloud import firestore
            credentials, detected_project = google.auth.default()
            client = firestore.Client(
                project=project or detected_project,
                credentials=credentials,
            )
            logger.info("Firestore connected via Application Default Credentials.")
            return client
    except Exception as exc:
        logger.warning(
            "Firestore connection failed (%s). "
            "Running in local-only mode — sessions and document persistence unavailable.",
            exc,
        )
        return None


def is_firestore_available() -> bool:
    return get_firestore_client() is not None


class FirestoreClient:
    """
    Firestore wrapper with no-op fallback when Firestore is unavailable.
    All write/read methods silently do nothing and return safe defaults.
    """

    def __init__(self) -> None:
        self._db = get_firestore_client()
        if self._db is None:
            logger.debug("FirestoreClient running in no-op mode.")

    @property
    def available(self) -> bool:
        return self._db is not None

    # ------------------------------------------------------------------ #
    # Generic CRUD
    # ------------------------------------------------------------------ #

    def set(self, collection: str, doc_id: str, data: Dict[str, Any]) -> str:
        if self._db:
            self._db.collection(collection).document(doc_id).set(data)
            logger.debug("SET %s/%s", collection, doc_id)
        return doc_id

    def merge(self, collection: str, doc_id: str, data: Dict[str, Any]) -> str:
        if self._db:
            self._db.collection(collection).document(doc_id).set(data, merge=True)
            logger.debug("MERGE %s/%s", collection, doc_id)
        return doc_id

    def get(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        if self._db:
            snap = self._db.collection(collection).document(doc_id).get()
            return snap.to_dict() if snap.exists else None
        return None

    def delete(self, collection: str, doc_id: str) -> None:
        if self._db:
            self._db.collection(collection).document(doc_id).delete()
            logger.debug("DELETE %s/%s", collection, doc_id)

    def list_all(self, collection: str) -> List[Dict[str, Any]]:
        if self._db:
            docs = self._db.collection(collection).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        return []

    def query(
        self,
        collection: str,
        filters: Optional[List[tuple]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self._db:
            return []
        ref = self._db.collection(collection)
        if filters:
            for field, op, value in filters:
                ref = ref.where(field, op, value)
        if order_by:
            ref = ref.order_by(order_by)
        if limit:
            ref = ref.limit(limit)
        return [{"id": d.id, **d.to_dict()} for d in ref.stream()]

    def batch_set(self, collection: str, items: List[Dict[str, Any]], id_field: str = "id") -> int:
        if not self._db:
            return 0
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
