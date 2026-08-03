"""Tests for the health endpoint and API-key authentication."""

from __future__ import annotations


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_requires_no_auth(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_protected_endpoint_rejects_missing_key(client):
    resp = client.post("/analyze-sentiment", json={"text": "hello"})
    assert resp.status_code == 401


def test_protected_endpoint_rejects_wrong_key(client):
    resp = client.post(
        "/analyze-sentiment",
        json={"text": "hello"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_dashboard_rejects_missing_key(client):
    resp = client.get("/dashboard/stats")
    assert resp.status_code == 401
