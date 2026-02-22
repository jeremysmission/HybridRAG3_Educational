# ============================================================================
# HybridRAG -- FastAPI Server Tests (tests/test_fastapi_server.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Tests all FastAPI REST API endpoints using FastAPI TestClient.
#   No live server needed -- TestClient runs everything in-process.
#
# USAGE:
#   pytest tests/test_fastapi_server.py -v
#
# INTERNET ACCESS: NONE
# ============================================================================

import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set environment before any imports
os.environ.setdefault("HYBRIDRAG_DATA_DIR", "D:\\RAG Indexed Data")
os.environ.setdefault("HYBRIDRAG_INDEX_FOLDER", "D:\\RAG Source Data")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from fastapi.testclient import TestClient
from src.api.server import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with lifespan context."""
    with TestClient(app) as c:
        yield c


# -------------------------------------------------------------------
# Health endpoint
# -------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_version(self, client):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


# -------------------------------------------------------------------
# Status endpoint
# -------------------------------------------------------------------

class TestStatus:
    def test_status_returns_200(self, client):
        r = client.get("/status")
        assert r.status_code == 200

    def test_status_has_chunk_count(self, client):
        r = client.get("/status")
        data = r.json()
        assert "chunk_count" in data
        assert isinstance(data["chunk_count"], int)
        assert data["chunk_count"] >= 0

    def test_status_has_mode(self, client):
        r = client.get("/status")
        data = r.json()
        assert data["mode"] in ("offline", "online")

    def test_status_has_source_count(self, client):
        r = client.get("/status")
        data = r.json()
        assert "source_count" in data
        assert isinstance(data["source_count"], int)


# -------------------------------------------------------------------
# Config endpoint
# -------------------------------------------------------------------

class TestConfig:
    def test_config_returns_200(self, client):
        r = client.get("/config")
        assert r.status_code == 200

    def test_config_has_embedding_model(self, client):
        r = client.get("/config")
        data = r.json()
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        assert data["embedding_dimension"] == 384

    def test_config_has_retrieval_settings(self, client):
        r = client.get("/config")
        data = r.json()
        assert "top_k" in data
        assert "min_score" in data
        assert "hybrid_search" in data


# -------------------------------------------------------------------
# Index status endpoint
# -------------------------------------------------------------------

class TestIndexStatus:
    def test_index_status_returns_200(self, client):
        r = client.get("/index/status")
        assert r.status_code == 200

    def test_index_status_not_active_by_default(self, client):
        r = client.get("/index/status")
        data = r.json()
        assert data["indexing_active"] is False


# -------------------------------------------------------------------
# Query endpoint
# -------------------------------------------------------------------

class TestQuery:
    def test_query_rejects_empty_question(self, client):
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422

    def test_query_rejects_missing_question(self, client):
        r = client.post("/query", json={})
        assert r.status_code == 422

    def test_query_accepts_valid_question(self, client):
        r = client.post("/query", json={"question": "What is HybridRAG?"})
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert "sources" in data
        assert "chunks_used" in data
        assert "latency_ms" in data


# -------------------------------------------------------------------
# Mode endpoint
# -------------------------------------------------------------------

class TestMode:
    def test_mode_rejects_invalid(self, client):
        r = client.put("/mode", json={"mode": "turbo"})
        assert r.status_code == 422

    def test_mode_switch_to_offline(self, client):
        r = client.put("/mode", json={"mode": "offline"})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "offline"


# -------------------------------------------------------------------
# Index start endpoint
# -------------------------------------------------------------------

class TestIndexStart:
    def test_index_rejects_bad_folder(self, client):
        r = client.post("/index", json={"source_folder": "/nonexistent/path"})
        assert r.status_code == 400
