"""
MongoDB integration for the Sentiment Analysis app.
Handles all database operations: dataset, predictions, feedback, metrics.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Always load from the same directory as this file so it works regardless of cwd
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(_ENV_PATH, override=True)

MONGO_URI     = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "sentiment_analysis")

_client = None
_db     = None


def get_db():
    """Return the database handle, creating the connection once."""
    global _client, _db
    if _db is not None:
        return _db

    if not MONGO_URI or MONGO_URI.strip() == "":
        raise ValueError(
            "MongoDB URI not configured. "
            "Edit backend/.env and set MONGO_URI to your Atlas connection string."
        )

    _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    # Ping to fail fast if credentials are wrong
    _client.admin.command("ping")
    _db = _client[MONGO_DB_NAME]
    print(f"[MongoDB] Connected to '{MONGO_DB_NAME}'")
    return _db


def is_connected():
    """Return True if the database is reachable."""
    try:
        get_db()
        return True
    except Exception:
        return False


# ── Collections ──────────────────────────────────────────────────────────────

def get_dataset_col():
    return get_db()["dataset"]

def get_predictions_col():
    return get_db()["predictions"]

def get_feedback_col():
    return get_db()["feedback"]

def get_metrics_col():
    return get_db()["model_metrics"]


# ── Dataset ───────────────────────────────────────────────────────────────────

def save_dataset(records: list[dict]):
    """
    Upsert all records into the dataset collection.
    Uses the review text as the unique key so duplicates are never inserted.
    """
    col = get_dataset_col()
    inserted = 0
    for rec in records:
        result = col.update_one(
            {"review": rec["review"]},          # match on text
            {"$setOnInsert": {                   # only insert, never overwrite
                "review":     rec["review"],
                "sentiment":  rec["sentiment"],
                "source":     rec.get("source", "sample"),
                "created_at": datetime.utcnow(),
            }},
            upsert=True,
        )
        if result.upserted_id:
            inserted += 1
    print(f"[MongoDB] dataset — {inserted} new rows inserted ({len(records)} total processed)")
    return inserted


def load_dataset() -> list[dict]:
    """Return all dataset records as plain dicts (no _id)."""
    col = get_dataset_col()
    return [
        {"review": d["review"], "sentiment": d["sentiment"]}
        for d in col.find({}, {"_id": 0, "review": 1, "sentiment": 1})
    ]


def add_review_to_dataset(review: str, sentiment: str, source: str = "analyzed"):
    """Add a single review if it doesn't already exist."""
    col = get_dataset_col()
    result = col.update_one(
        {"review": review},
        {"$setOnInsert": {
            "review":     review,
            "sentiment":  sentiment,
            "source":     source,
            "created_at": datetime.utcnow(),
        }},
        upsert=True,
    )
    return result.upserted_id is not None   # True = newly inserted


def get_dataset_count() -> int:
    return get_dataset_col().count_documents({})


def get_sentiment_distribution() -> dict:
    pipeline = [
        {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}}
    ]
    return {doc["_id"]: doc["count"] for doc in get_dataset_col().aggregate(pipeline)}


def get_unretrained_count() -> int:
    """Return the number of dataset rows added since the last retrain."""
    return get_dataset_col().count_documents({"retrained": {"$ne": True}})


def mark_all_retrained():
    """Mark all dataset rows as included in the latest retrain."""
    get_dataset_col().update_many(
        {"retrained": {"$ne": True}},
        {"$set": {"retrained": True}},
    )


# ── Predictions ───────────────────────────────────────────────────────────────

def save_prediction(text: str, sentiment: str, confidence: float, probabilities: dict):
    """Persist one prediction record."""
    col = get_predictions_col()
    col.insert_one({
        "text":          text,
        "sentiment":     sentiment,
        "confidence":    confidence,
        "probabilities": probabilities,
        "created_at":    datetime.utcnow(),
    })


def get_total_predictions() -> int:
    return get_predictions_col().count_documents({})


def get_recent_predictions(limit: int = 20) -> list[dict]:
    col = get_predictions_col()
    return [
        {
            "text":       d["text"],
            "sentiment":  d["sentiment"],
            "confidence": d["confidence"],
            "created_at": d["created_at"].isoformat(),
        }
        for d in col.find({}, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
    ]


# ── Feedback ──────────────────────────────────────────────────────────────────

def save_feedback(text: str, predicted: str, actual: str, confidence: float):
    """Persist one feedback record."""
    col = get_feedback_col()
    col.insert_one({
        "text":                text,
        "predicted_sentiment": predicted,
        "actual_sentiment":    actual,
        "confidence":          confidence,
        "created_at":          datetime.utcnow(),
    })


def load_pending_feedback() -> list[dict]:
    """Return all feedback not yet used for retraining."""
    col = get_feedback_col()
    return [
        {
            "text":                d["text"],
            "predicted_sentiment": d["predicted_sentiment"],
            "actual_sentiment":    d["actual_sentiment"],
            "confidence":          d["confidence"],
        }
        for d in col.find({"used_for_training": {"$ne": True}}, {"_id": 0})
    ]


def mark_feedback_used():
    """Mark all pending feedback as consumed after retraining."""
    get_feedback_col().update_many(
        {"used_for_training": {"$ne": True}},
        {"$set": {"used_for_training": True}},
    )


def get_feedback_count() -> int:
    return get_feedback_col().count_documents({"used_for_training": {"$ne": True}})


# ── Model Metrics ─────────────────────────────────────────────────────────────

def save_metrics(metrics: dict):
    """Insert a new metrics snapshot (keeps history of every retrain)."""
    col = get_metrics_col()
    col.insert_one({**metrics, "created_at": datetime.utcnow()})


def load_latest_metrics() -> dict:
    """Return the most recent metrics document (without _id)."""
    col = get_metrics_col()
    doc = col.find_one({}, {"_id": 0}, sort=[("created_at", DESCENDING)])
    return doc or {}
