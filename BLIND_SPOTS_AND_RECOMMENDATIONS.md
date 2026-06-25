# Catalyst Detection — What Shipped

_Auto 52-Week High/Low Analysis — June 2026. Scope: thematic + stock-specific catalysts only._

## What's already strong

Detection of the move itself is complete: `engine.py` flags any stock touching its 52-week high/low regardless of cause. Attribution pulls news from Tavily, NewsAPI, Exa and yfinance in parallel (`_fetch_all_news_items`), the LLM deep-dive reads all of it, and `sector_breadth` / `diagnose_move` label sector-wide moves. The weak link was the deterministic layer (`analysis_utils.py`), which used context-blind keyword lists.

## Shipped in this pass

Focused on sharpening **stock-specific** and **thematic** attribution. Three changes:

**1. Context-aware sentiment guard (stock-specific).** `_sentiment_ok` scored single words only, so "tariff", "cut", "ban", "probe" were always negative — a bullish *tariff cut* / *PLI scheme* / *cleared of probe* headline got rejected as the driver of a 52-week high (the original textile-tariff problem). It is now phrase-aware: multi-word phrases (`tariff cut`, `cuts duty`, `zero duty`, `ban lifted`, `cleared of`, `rating upgrade`, …) are scored first and weighted x2; genuine negatives (`tariff hike`, `export ban`, `profit warning`) are still rejected. Single words now match on word boundaries (so "win" no longer matches "winter"), and the catalyst keyword lists were expanded (QIP, fundraise, demerger, commissioning, rating actions, etc.). Optional LLM sentiment+entity classifier added as `llm_relevance_check` — used by the deep-dive origin pick when `POLICY_LLM_GUARD=1`, with the keyword guard as automatic fallback. Verified against headline test cases (7/7).

**2. Thematic breadth (`themes.py`).** Breadth previously grouped only by exact yfinance `sector`, so cross-sector themes (defence, railways, EV, solar, PSU, China+1, sugar, textiles, capital markets, real estate, infra) got diluted. `themes.theme_breadth()` checks how many names from a stock's theme(s) also hit a 52-week extreme in the same screen and feeds a "Thematic move" driver into `diagnose_move`, which now classifies these as "Thematic high/low". Also `normalise_sector()` collapses the NSE-feed and yfinance sector taxonomies into one canonical set. Verified (Defence basket detected; non-theme names return None).

**3. Exchange filings as a stock-specific news source (`exchange_filings.py`).** Many Indian mid/small-cap catalysts break first as NSE/BSE corporate announcements, ahead of news sites. `fetch_filings()` pulls recent announcements for a ticker (NSE cookie session, BSE fallback) in the same shape as the other news sources and is merged into `_fetch_all_news_items` so they flow straight into catalyst selection. Degrades to `[]` if the exchange APIs are unreachable.

## Deliberately not pursued

Per the decision to keep the app focused and avoid speculative narrative ("AI slop"), these were considered and intentionally left out: a trade/policy data feed, passive-flow (index/MSCI/F&O) tracking, corporate-action ex-date flags, global/commodity→sector linkage drivers, and block/bulk-deal flow. They attribute moves to inferred macro causes rather than confirmed stock-specific or thematic catalysts, which adds noise more than accuracy.
