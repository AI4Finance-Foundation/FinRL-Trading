# Adanos Market Sentiment Features

FinRL-X can optionally enrich offline US equity datasets with daily market
sentiment from [Adanos](https://api.adanos.org/docs). The processor keeps API
access outside strategy and environment steps, caches successful responses
locally, and returns one row per ticker and UTC date.

Set your Adanos key in the `ADANOS_API_KEY` environment variable and fetch only
the sources needed by the experiment:

```python
from src.data.adanos_sentiment import AdanosSentimentProcessor

processor = AdanosSentimentProcessor()
features = processor.fetch_daily_features(
    tickers=["AAPL", "MSFT", "NVDA"],
    start_date="2026-07-01",
    end_date="2026-07-30",
    sources=("reddit", "x", "news", "polymarket"),
)

prices_with_sentiment = processor.merge_with_prices(price_data, features)
```

The returned columns are prefixed by source, for example
`adanos_reddit_sentiment_score`, `adanos_news_buzz_score`, and
`adanos_polymarket_activity_count`. Missing observations remain null rather
than being converted to neutral sentiment or zero activity.

Requests use inclusive `from` and `to` UTC dates. Choose a window supported by
the API key's plan and the selected source. Completed UTC windows are cached as
JSON under `data/cache/adanos` by default; windows that include the current UTC
date are fetched again because their aggregates are still changing. Pass
`force_refresh=True` to refresh a completed window. The API key is sent only in
the `X-API-Key` request header and is never written to cache.

Sentiment features are alternative research data, not trading recommendations.
Daily aggregates cover their full UTC date. For decisions made before that date
has ended, lag the features to the next eligible trading timestamp; the exact
date join shown above does not enforce that causal delay. Preserve the dataset's
existing train/test boundaries as well.
