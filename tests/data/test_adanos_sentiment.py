import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data.adanos_sentiment import AdanosSentimentProcessor


@pytest.fixture(autouse=True)
def adanos_api_key(monkeypatch):
    monkeypatch.setenv("ADANOS_API_KEY", "placeholder")


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(next(self.payloads))


def test_fetch_daily_features_uses_explicit_utc_window_and_pivots_sources(tmp_path):
    session = FakeSession(
        [
            {
                "ticker": "AAPL",
                "daily_trend": [
                    {
                        "date": "2026-07-02",
                        "mentions": 18,
                        "sentiment_score": 0.4,
                        "buzz_score": 52.5,
                        "bullish_pct": 60,
                        "bearish_pct": 20,
                    },
                    {
                        "date": "2026-07-01",
                        "mentions": 10,
                        "sentiment_score": 0.2,
                        "buzz_score": 40.0,
                        "bullish_pct": 50,
                        "bearish_pct": 10,
                    },
                ],
            },
            {
                "ticker": "AAPL",
                "daily_trend": [
                    {
                        "date": "2026-07-02",
                        "trade_count": 7,
                        "sentiment_score": -0.1,
                        "buzz_score": 31.0,
                        "bullish_pct": 30,
                        "bearish_pct": 45,
                    }
                ],
            },
        ]
    )
    processor = AdanosSentimentProcessor(cache_dir=str(tmp_path), session=session)

    result = processor.fetch_daily_features(
        ["$aapl", "AAPL"],
        "2026-07-01",
        "2026-07-02",
        sources=("reddit", "polymarket"),
    )

    assert list(result["tic"]) == ["AAPL", "AAPL"]
    assert list(result["datadate"]) == [
        pd.Timestamp("2026-07-01"),
        pd.Timestamp("2026-07-02"),
    ]
    assert result.loc[1, "adanos_reddit_activity_count"] == 18
    assert result.loc[1, "adanos_polymarket_activity_count"] == 7
    assert pd.isna(result.loc[0, "adanos_polymarket_sentiment_score"])

    assert len(session.calls) == 2
    for _, kwargs in session.calls:
        assert kwargs["headers"] == {"X-API-Key": "placeholder"}
        assert kwargs["params"] == {"from": "2026-07-01", "to": "2026-07-02"}
        assert "days" not in kwargs["params"]


def test_fetch_daily_features_reuses_successful_cached_response(tmp_path):
    payload = {
        "ticker": "MSFT",
        "daily_trend": [
            {
                "date": "2026-07-01",
                "mentions": 4,
                "sentiment_score": 0.1,
                "buzz_score": 12.0,
                "bullish_pct": 40,
                "bearish_pct": 15,
            }
        ],
    }
    session = FakeSession([payload])
    processor = AdanosSentimentProcessor(cache_dir=str(tmp_path), session=session)

    first = processor.fetch_daily_features(
        ["MSFT"], "2026-07-01", "2026-07-01", sources=("news",)
    )
    second = processor.fetch_daily_features(
        ["MSFT"], "2026-07-01", "2026-07-01", sources=("news",)
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(session.calls) == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    assert "placeholder" not in cache_files[0].read_text()
    assert json.loads(cache_files[0].read_text()) == payload


def test_fetch_daily_features_does_not_cache_current_utc_day(tmp_path):
    current_date = datetime.now(timezone.utc).date().isoformat()
    payload = {
        "ticker": "MSFT",
        "daily_trend": [{"date": current_date, "mentions": 2}],
    }
    session = FakeSession([payload, payload])
    processor = AdanosSentimentProcessor(cache_dir=str(tmp_path), session=session)

    processor.fetch_daily_features(
        ["MSFT"], current_date, current_date, sources=("reddit",)
    )
    processor.fetch_daily_features(
        ["MSFT"], current_date, current_date, sources=("reddit",)
    )

    assert len(session.calls) == 2
    assert list(tmp_path.glob("*.json")) == []


def test_merge_with_prices_keeps_missing_sentiment_distinct_from_neutral():
    prices = pd.DataFrame(
        {
            "tic": ["aapl", "AAPL"],
            "datadate": ["2026-07-01T23:00:00-04:00", "2026-07-03"],
            "adj_close": [200.0, 202.0],
        }
    )
    features = pd.DataFrame(
        {
            "tic": ["AAPL"],
            "datadate": [pd.Timestamp("2026-07-02")],
            "adanos_news_sentiment_score": [0.0],
        }
    )

    result = AdanosSentimentProcessor.merge_with_prices(prices, features)

    assert result.loc[0, "datadate"] == pd.Timestamp("2026-07-02")
    assert result.loc[0, "adanos_news_sentiment_score"] == 0.0
    assert pd.isna(result.loc[1, "adanos_news_sentiment_score"])


@pytest.mark.parametrize(
    ("start_date", "end_date", "sources", "message"),
    [
        ("2026-07-02", "2026-07-01", ("reddit",), "start_date"),
        ("not-a-date", "2026-07-01", ("reddit",), "valid YYYY-MM-DD"),
        ("2026-07-01", "2026-07-02", ("unknown",), "Unsupported"),
    ],
)
def test_fetch_daily_features_rejects_invalid_requests(
    tmp_path, start_date, end_date, sources, message
):
    processor = AdanosSentimentProcessor(
        cache_dir=str(tmp_path), session=FakeSession([])
    )

    with pytest.raises(ValueError, match=message):
        processor.fetch_daily_features(["AAPL"], start_date, end_date, sources)


def test_api_key_is_required(monkeypatch, tmp_path):
    monkeypatch.delenv("ADANOS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="ADANOS_API_KEY"):
        AdanosSentimentProcessor(cache_dir=str(tmp_path))
