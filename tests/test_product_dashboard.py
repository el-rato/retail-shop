"""Tests for product classification and dashboard aggregation."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.database.models import Customer, Visit
from app.services import dashboard_service, product_service


class TestProductClassification:
    def _stub_classifier(self, monkeypatch):
        classifier = MagicMock()
        classifier.classify.return_value = [
            {"label": "sneaker", "category": "sneaker", "confidence": 0.93},
            {"label": "sandal", "category": "sandal", "confidence": 0.05},
            {"label": "bag", "category": "bag", "confidence": 0.02},
        ]
        fake_pipeline = MagicMock()
        fake_pipeline.product = classifier
        monkeypatch.setattr(product_service, "get_pipeline", lambda: fake_pipeline)
        return classifier

    def test_classify_product_service(self, db, monkeypatch):
        self._stub_classifier(monkeypatch)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = product_service.classify_product(db, image)
        assert result["top_prediction"]["label"] == "sneaker"
        assert len(result["all_predictions"]) == 3

    def test_classify_product_logs_prediction(self, db, monkeypatch):
        self._stub_classifier(monkeypatch)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        product_service.classify_product(db, image)
        from app.database.models import ProductPrediction

        assert db.query(ProductPrediction).count() == 1


class TestDashboard:
    def test_empty_stats(self, db):
        stats = dashboard_service.build_dashboard_stats(db)
        assert stats["total_customers"] == 0
        assert stats["total_visits"] == 0
        assert stats["top_customer"] is None

    def test_stats_with_data(self, db):
        customer = Customer(name="Alice", face_encodings=[])
        db.add(customer)
        db.commit()
        db.refresh(customer)
        db.add(Visit(customer_id=customer.id, confidence=0.9, location="Downtown"))
        db.commit()

        stats = dashboard_service.build_dashboard_stats(db)
        assert stats["total_customers"] == 1
        assert stats["total_visits"] == 1
        assert stats["unique_visitors"] == 1
        assert stats["top_customer"]["name"] == "Alice"
        assert len(stats["recent_visits"]) == 1

    def test_dashboard_endpoint(self, client, auth_headers):
        resp = client.get("/dashboard/stats", headers=auth_headers)
        assert resp.status_code == 200
        body = resp.json()
        for key in ("total_customers", "sentiment_counts", "recent_visits", "generated_at"):
            assert key in body
