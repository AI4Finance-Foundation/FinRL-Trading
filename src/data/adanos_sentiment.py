"""Optional Adanos market sentiment features for FinRL datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import pandas as pd
import requests

SOURCE_ENDPOINTS = {
    "reddit": "/reddit/stocks/v1/stock/{ticker}",
    "x": "/x/stocks/v1/stock/{ticker}",
    "news": "/news/stocks/v1/stock/{ticker}",
    "polymarket": "/polymarket/stocks/v1/stock/{ticker}",
}
SOURCE_ACTIVITY_FIELDS = {
    "reddit": "mentions",
    "x": "mentions",
    "news": "mentions",
    "polymarket": "trade_count",
}
DAILY_METRICS = (
    "sentiment_score",
    "buzz_score",
    "bullish_pct",
    "bearish_pct",
    "activity_count",
)


class AdanosSentimentProcessor:
    """Fetch and prepare daily Adanos sentiment data for offline research."""

    def __init__(
        self,
        base_url: str = "https://api.adanos.org",
        cache_dir: str = "./data/cache/adanos",
        timeout: int = 30,
        session: Optional[requests.Session] = None,
    ) -> None:
        access_value = os.getenv("ADANOS_API_KEY")
        if not access_value:
            raise ValueError("Set ADANOS_API_KEY before using Adanos sentiment data.")

        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.session = session or requests.Session()
        self._request_headers = dict.fromkeys(("X-API-Key",), access_value)

    def fetch_daily_features(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        sources: Sequence[str] = tuple(SOURCE_ENDPOINTS),
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return one row per ticker and UTC date with source-prefixed features."""
        start = self._parse_date(start_date, "start_date")
        end = self._parse_date(end_date, "end_date")
        if start > end:
            raise ValueError("start_date must be on or before end_date")

        normalized_tickers = self._normalize_tickers(tickers)
        normalized_sources = self._normalize_sources(sources)
        frames = []

        for source in normalized_sources:
            records = []
            for ticker in normalized_tickers:
                payload = self._get_payload(
                    source, ticker, start, end, force_refresh=force_refresh
                )
                records.extend(self._daily_records(source, ticker, payload, start, end))

            columns = ["tic", "datadate"] + self._feature_columns(source)
            frame = pd.DataFrame.from_records(records, columns=columns)
            if not frame.empty:
                duplicates = frame.duplicated(["tic", "datadate"], keep=False)
                if duplicates.any():
                    raise ValueError(
                        f"Adanos returned duplicate {source} observations for a ticker/date"
                    )
            frames.append(frame)

        result = frames[0]
        for frame in frames[1:]:
            result = result.merge(
                frame, on=["tic", "datadate"], how="outer", validate="one_to_one"
            )

        if result.empty:
            return result
        return result.sort_values(["tic", "datadate"]).reset_index(drop=True)

    @staticmethod
    def merge_with_prices(
        price_data: pd.DataFrame,
        sentiment_features: pd.DataFrame,
        ticker_column: str = "tic",
        date_column: str = "datadate",
    ) -> pd.DataFrame:
        """Left-join daily sentiment features without filling missing observations."""
        required = {ticker_column, date_column}
        missing_prices = required.difference(price_data.columns)
        missing_features = required.difference(sentiment_features.columns)
        if missing_prices:
            raise ValueError(
                f"price_data is missing required columns: {sorted(missing_prices)}"
            )
        if missing_features:
            raise ValueError(
                "sentiment_features is missing required columns: "
                f"{sorted(missing_features)}"
            )

        prices = price_data.copy()
        features = sentiment_features.copy()
        for frame in (prices, features):
            frame[ticker_column] = (
                frame[ticker_column].astype(str).str.strip().str.lstrip("$").str.upper()
            )
            frame[date_column] = (
                pd.to_datetime(frame[date_column], format="mixed", utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )

        if features.duplicated([ticker_column, date_column]).any():
            raise ValueError("sentiment_features must contain one row per ticker/date")

        return prices.merge(
            features,
            on=[ticker_column, date_column],
            how="left",
            validate="many_to_one",
        )

    def _get_payload(
        self,
        source: str,
        ticker: str,
        start: date,
        end: date,
        force_refresh: bool,
    ) -> Mapping[str, object]:
        cache_path = self._cache_path(source, ticker, start, end)
        cacheable = end < datetime.now(timezone.utc).date()
        if cacheable and cache_path.exists() and not force_refresh:
            with cache_path.open(encoding="utf-8") as cache_file:
                payload = json.load(cache_file)
            return self._validate_payload(payload)

        endpoint = SOURCE_ENDPOINTS[source].format(ticker=ticker)
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            headers=self._request_headers,
            params={"from": start.isoformat(), "to": end.isoformat()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = self._validate_payload(response.json())
        if cacheable:
            self._write_cache(cache_path, payload)
        return payload

    def _cache_path(self, source: str, ticker: str, start: date, end: date) -> Path:
        request_identity = json.dumps(
            {
                "base_url": self.base_url,
                "source": source,
                "ticker": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(request_identity).hexdigest()[:16]
        return self.cache_dir / f"{source}-{digest}.json"

    def _write_cache(self, cache_path: Path, payload: Mapping[str, object]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                delete=False,
            ) as cache_file:
                json.dump(payload, cache_file)
                temporary_path = Path(cache_file.name)
            temporary_path.replace(cache_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _daily_records(
        source: str,
        ticker: str,
        payload: Mapping[str, object],
        start: date,
        end: date,
    ) -> list[dict[str, object]]:
        daily_trend = payload.get("daily_trend") or []
        if not isinstance(daily_trend, list):
            raise ValueError("Adanos response field 'daily_trend' must be a list")

        activity_field = SOURCE_ACTIVITY_FIELDS[source]
        records = []
        for item in daily_trend:
            if not isinstance(item, Mapping):
                raise ValueError("Adanos daily_trend items must be objects")
            observation_date = AdanosSentimentProcessor._parse_date(
                item.get("date"), "daily_trend.date"
            )
            if observation_date < start or observation_date > end:
                continue
            prefix = f"adanos_{source}_"
            records.append(
                {
                    "tic": ticker,
                    "datadate": pd.Timestamp(observation_date),
                    f"{prefix}sentiment_score": item.get("sentiment_score"),
                    f"{prefix}buzz_score": item.get("buzz_score"),
                    f"{prefix}bullish_pct": item.get("bullish_pct"),
                    f"{prefix}bearish_pct": item.get("bearish_pct"),
                    f"{prefix}activity_count": item.get(activity_field),
                }
            )
        return records

    @staticmethod
    def _parse_date(value: object, field_name: str) -> date:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a YYYY-MM-DD string")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a valid YYYY-MM-DD date") from exc

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
        normalized = []
        for ticker in tickers:
            value = str(ticker).strip().lstrip("$").upper()
            if value and value not in normalized:
                normalized.append(value)
        if not normalized:
            raise ValueError("At least one ticker is required")
        return normalized

    @staticmethod
    def _normalize_sources(sources: Sequence[str]) -> list[str]:
        normalized = [source.lower() for source in sources]
        if not normalized:
            raise ValueError("At least one Adanos source is required")
        unsupported = set(normalized).difference(SOURCE_ENDPOINTS)
        if unsupported:
            raise ValueError(f"Unsupported Adanos sources: {sorted(unsupported)}")
        return list(dict.fromkeys(normalized))

    @staticmethod
    def _feature_columns(source: str) -> list[str]:
        return [f"adanos_{source}_{metric}" for metric in DAILY_METRICS]

    @staticmethod
    def _validate_payload(payload: object) -> Mapping[str, object]:
        if not isinstance(payload, Mapping):
            raise ValueError("Adanos response must be a JSON object")
        return payload
