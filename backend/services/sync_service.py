import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Any
from sqlalchemy.orm import Session
from backend.db.postgres import StockProfile, StockData, StockNewsItem, StockInsightSnapshot, Prediction, ModelMetrics, StockFinancial
from backend.services.inference_pipeline import run_training_pipeline

logger = logging.getLogger(__name__)

def _safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A"): return None
    try: return float(value)
    except (TypeError, ValueError): return None

def _is_stale(updated_at: datetime | None, ttl_seconds: int) -> bool:
    if updated_at is None: return True
    return updated_at <= datetime.utcnow() - timedelta(seconds=ttl_seconds)

def full_stock_sync(db: Session, symbol: str) -> None:
    """Deep sync for a single stock: snapshot, metadata, history, and news."""
    try:
        stock = db.query(StockProfile).filter(StockProfile.symbol == symbol).first()
        if not stock:
            return

        ticker = yf.Ticker(symbol)
        
        # 1. Update basic price snapshot
        _refresh_stock_snapshot(stock, ticker)
        
        # 2. Update metadata (Sector, Industry, CEO, etc.)
        _refresh_stock_metadata(stock, ticker)
        
        # 3. Update History (Charts)
        _refresh_stock_history(db, stock, ticker)
        
        # 4. Update News
        _refresh_stock_news(db, stock, ticker)
        
        # 5. Update Financials (Revenue & Profit)
        _refresh_stock_financials(db, stock, ticker)
        
        # 6. Trigger AI Model Training
        try:
            logger.info(f"🤖 Triggering AI Training for {symbol}...")
            run_training_pipeline(symbol)
        except Exception as e:
            logger.warning(f"AI Training skipped for {symbol}: {e}")
            
        stock.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"✓ Deep sync completed for {symbol}")
    except Exception as exc:
        db.rollback()
        logger.warning(f"Deep sync failed for {symbol}: {exc}")

def _refresh_stock_snapshot(stock: StockProfile, ticker: Any) -> None:
    try:
        fast_info = getattr(ticker, "fast_info", None)
        price = _safe_float(getattr(fast_info, "last_price", None))
        prev_close = _safe_float(getattr(fast_info, "previous_close", None))
        
        if price is None:
            info = ticker.info or {}
            price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
            prev_close = prev_close or _safe_float(info.get("previousClose"))

        if price is not None:
            stock.price = round(price, 2)
            if prev_close and prev_close > 0:
                stock.change = round(price - prev_close, 2)
                stock.change_pct = round(((price - prev_close) / prev_close) * 100, 2)
    except Exception:
        pass

def _refresh_stock_metadata(stock: StockProfile, ticker: Any) -> None:
    try:
        info = ticker.info or {}
        stock.name = info.get("longName") or stock.name
        stock.sector = info.get("sector") or stock.sector
        stock.industry = info.get("industry") or stock.industry
        stock.description = info.get("longBusinessSummary") or stock.description
        stock.ceo = _extract_ceo(info) or stock.ceo
        stock.market_cap = _safe_float(info.get("marketCap")) or stock.market_cap
        stock.pe_ratio = _safe_float(info.get("trailingPE")) or stock.pe_ratio
        stock.eps = _safe_float(info.get("trailingEps")) or stock.eps
        stock.avg_volume = _safe_float(info.get("averageVolume")) or stock.avg_volume
        stock.week_52_low = _safe_float(info.get("fiftyTwoWeekLow")) or stock.week_52_low
        stock.week_52_high = _safe_float(info.get("fiftyTwoWeekHigh")) or stock.week_52_high
    except Exception:
        pass

def _extract_ceo(info: dict) -> str | None:
    officers = info.get("companyOfficers") or []
    for o in officers:
        if "ceo" in str(o.get("title", "")).lower():
            return o.get("name")
    return None

def _refresh_stock_history(db: Session, stock: StockProfile, ticker: Any) -> None:
    try:
        hist = ticker.history(period="1y", interval="1d")
        if hist is None or hist.empty: return

        for ts, row in hist.iterrows():
            dt = ts.to_pydatetime().replace(tzinfo=None)
            # Simple upsert logic
            existing = db.query(StockData).filter(StockData.symbol == stock.symbol, StockData.date == dt).first()
            if not existing:
                db.add(StockData(
                    symbol=stock.symbol,
                    date=dt,
                    open=_safe_float(row.get("Open")),
                    high=_safe_float(row.get("High")),
                    low=_safe_float(row.get("Low")),
                    close=_safe_float(row.get("Close")),
                    volume=_safe_float(row.get("Volume"))
                ))
    except Exception:
        pass

def _refresh_stock_financials(db: Session, stock: StockProfile, ticker: Any) -> None:
    """Fetch income statement data for the Revenue/Profit graphs."""
    try:
        # Fetch quarterly financials
        fin = ticker.quarterly_financials
        if fin is None or fin.empty:
            fin = ticker.financials # Fallback to annual
            
        if fin is not None and not fin.empty:
            # yfinance returns metrics as index, dates as columns
            revenue_row = fin.loc['Total Revenue'] if 'Total Revenue' in fin.index else None
            profit_row = fin.loc['Net Income'] if 'Net Income' in fin.index else None
            
            # Save the 4 most recent periods
            cols = fin.columns[:4]
            for col in cols:
                dt = col.to_pydatetime().replace(tzinfo=None)
                # Label like "Q1 2024" or just the year
                label = f"Q{ (dt.month-1)//3 + 1 } {dt.year}"
                
                rev_val = _safe_float(revenue_row[col]) if revenue_row is not None else None
                prof_val = _safe_float(profit_row[col]) if profit_row is not None else None
                
                # Simple upsert
                existing = db.query(StockFinancial).filter(
                    StockFinancial.symbol == stock.symbol,
                    StockFinancial.date == dt
                ).first()
                
                if not existing:
                    db.add(StockFinancial(
                        symbol=stock.symbol,
                        date=dt,
                        period_label=label,
                        revenue=rev_val,
                        net_income=prof_val,
                        is_quarterly=True
                    ))
                else:
                    existing.revenue = rev_val
                    existing.net_income = prof_val
                    
            # Update summary fields on StockProfile too
            if revenue_row is not None:
                stock.revenue = _safe_float(revenue_row.iloc[0])
            if profit_row is not None:
                stock.net_income = _safe_float(profit_row.iloc[0])
                
        db.commit()
    except Exception as e:
        logger.warning(f"Financial sync failed for {stock.symbol}: {e}")

def _refresh_stock_news(db: Session, stock: StockProfile, ticker: Any) -> None:
    try:
        news = getattr(ticker, "news", []) or []
        for item in news[:5]:
            title = item.get("title")
            if not title: continue
            existing = db.query(StockNewsItem).filter(StockNewsItem.symbol == stock.symbol, StockNewsItem.title == title).first()
            if not existing:
                db.add(StockNewsItem(
                    symbol=stock.symbol,
                    title=title,
                    publisher=item.get("publisher", "Yahoo Finance"),
                    link=item.get("link", ""),
                    published_at=datetime.fromtimestamp(item.get("providerPublishTime", datetime.now().timestamp()))
                ))
    except Exception:
        pass
