import csv
import io
import logging
from typing import Any

import yfinance as yf
from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.orm import Session

from backend.db.postgres import (
    ModelMetrics,
    Prediction,
    StockData,
    StockProfile,
    get_db,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["Data IO"])

DEFAULT_DASHBOARD_INDICES = [
    {"name": "Nasdaq", "value": "16,340.12", "chg": "+1.24%", "up": True},
    {"name": "Dow 30", "value": "39,112.16", "chg": "+0.42%", "up": True},
    {"name": "S&P 500", "value": "5,248.49", "chg": "+0.87%", "up": True},
    {"name": "Gold", "value": "2,342.10", "chg": "+0.12%", "up": True},
    {"name": "Bitcoin", "value": "68,432.22", "chg": "+2.45%", "up": True},
]

KNOWN_FOUNDED = {
    "AAPL": "1976", "MSFT": "1975", "GOOGL": "1998", "AMZN": "1994",
    "META": "2004", "NVDA": "1993", "TSLA": "2003", "NFLX": "1997",
    "AMD": "1969", "INTC": "1968", "CRM": "1999", "UBER": "2009",
}


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fast_info_value(fast_info: Any, key: str) -> Any:
    if fast_info is None:
        return None
    try:
        return fast_info[key]
    except Exception:
        return getattr(fast_info, key, None)


def _format_market_cap(value: float | None) -> str:
    if not value:
        return "N/A"
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def _format_volume(value: float | None) -> str:
    if not value:
        return "N/A"
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{int(value)}"


def _format_ratio(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}{suffix}"


def _format_currency(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"${value:.2f}"


def _extract_ceo_name(info: dict[str, Any]) -> str | None:
    officers = info.get("companyOfficers") or []
    ceo_name = None
    for officer in officers:
        title = str(officer.get("title", "")).lower()
        if "ceo" in title or "chief executive" in title:
            ceo_name = officer.get("name")
            break
    if not ceo_name and officers:
        ceo_name = officers[0].get("name")
    if not ceo_name:
        return None
    for prefix in ["Mr. ", "Ms. ", "Mrs. ", "Dr. ", "Sir "]:
        ceo_name = ceo_name.replace(prefix, "")
    return ceo_name


def _serialize_stock_summary(stock: StockProfile) -> dict[str, Any]:
    return {
        "id": stock.id,
        "sym": stock.symbol,
        "symbol": stock.symbol,
        "name": stock.name,
        "sector": stock.sector or "Other",
        "industry": stock.industry or "—",
        "country": stock.country or "—",
        "price": round(stock.price or 0.0, 2),
        "chg": round(stock.change or 0.0, 2),
        "chgPct": round(stock.change_pct or 0.0, 2),
        "marketCap": _format_market_cap(stock.market_cap),
        "marketCapValue": stock.market_cap,
        "isActive": bool(stock.is_active),
    }


def _serialize_stock_fundamentals(stock: StockProfile, history: list[float]) -> dict[str, Any]:
    return {
        "symbol": stock.symbol,
        "name": stock.name,
        "mktcap": _format_market_cap(stock.market_cap),
        "pe": _format_ratio(stock.pe_ratio, "x"),
        "eps": _format_currency(stock.eps),
        "vol": _format_volume(stock.avg_volume),
        "low": _format_currency(stock.week_52_low),
        "high": _format_currency(stock.week_52_high),
        "beta": _format_ratio(stock.beta),
        "div": f"{stock.dividend_yield:.2f}%" if stock.dividend_yield is not None else "None",
        "desc": stock.description or "Description not available.",
        "ceo": stock.ceo or "N/A",
        "founded": stock.founded or KNOWN_FOUNDED.get(stock.symbol, "N/A"),
        "sector": stock.sector or "N/A",
        "industry": stock.industry or "N/A",
        "country": stock.country or "N/A",
        "employees": f"{stock.employees:,}" if stock.employees else "N/A",
        "price": round(stock.price or 0.0, 2),
        "history": history,
    }


def _get_active_stocks(db: Session) -> list[StockProfile]:
    return (
        db.query(StockProfile)
        .filter(StockProfile.is_active.is_(True))
        .order_by(StockProfile.symbol.asc())
        .all()
    )


def _get_ticker_map(symbols: list[str]) -> dict[str, Any]:
    if not symbols:
        return {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
        return getattr(tickers, "tickers", {}) or {}
    except Exception as exc:
        logger.warning("Batch ticker fetch failed: %s", exc)
        return {}


def _refresh_stock_snapshot(stock: StockProfile, ticker: Any | None = None) -> None:
    if ticker is None:
        ticker = yf.Ticker(stock.symbol)

    try:
        fast_info = getattr(ticker, "fast_info", None)
        price = _safe_float(_fast_info_value(fast_info, "last_price"))
        prev_close = _safe_float(_fast_info_value(fast_info, "previous_close"))
        volume = _safe_float(_fast_info_value(fast_info, "last_volume")) or _safe_float(_fast_info_value(fast_info, "ten_day_average_volume"))

        if price is not None:
            stock.price = round(price, 2)
        if prev_close is not None:
            stock.previous_close = round(prev_close, 2)
        if price is not None and prev_close not in (None, 0):
            stock.change = round(price - prev_close, 2)
            stock.change_pct = round(((price - prev_close) / prev_close) * 100, 2)
        if volume is not None:
            stock.avg_volume = volume
    except Exception as exc:
        logger.warning("Quote refresh failed for %s: %s", stock.symbol, exc)


def _refresh_stock_metadata(stock: StockProfile, ticker: Any | None = None) -> None:
    if ticker is None:
        ticker = yf.Ticker(stock.symbol)

    try:
        info = getattr(ticker, "info", None) or {}
        if not info:
            return

        stock.name = stock.name or info.get("shortName") or info.get("longName") or stock.symbol
        stock.sector = stock.sector or info.get("sector")
        stock.industry = stock.industry or info.get("industry")
        stock.description = stock.description or info.get("longBusinessSummary")
        stock.ceo = stock.ceo or _extract_ceo_name(info)
        stock.founded = stock.founded or KNOWN_FOUNDED.get(stock.symbol) or str(info.get("ipoExpectedDate") or "") or None
        stock.country = stock.country or info.get("country")
        stock.employees = stock.employees or _safe_int(info.get("fullTimeEmployees"))

        market_cap = _safe_float(info.get("marketCap"))
        pe_ratio = _safe_float(info.get("trailingPE"))
        eps = _safe_float(info.get("trailingEps"))
        week_low = _safe_float(info.get("fiftyTwoWeekLow"))
        week_high = _safe_float(info.get("fiftyTwoWeekHigh"))
        beta = _safe_float(info.get("beta"))
        dividend_yield = _safe_float(info.get("dividendYield"))
        volume = _safe_float(info.get("volume"))
        price = _safe_float(info.get("currentPrice")) or _safe_float(info.get("regularMarketPrice"))

        if market_cap is not None:
            stock.market_cap = market_cap
        if pe_ratio is not None:
            stock.pe_ratio = pe_ratio
        if eps is not None:
            stock.eps = eps
        if week_low is not None:
            stock.week_52_low = week_low
        if week_high is not None:
            stock.week_52_high = week_high
        if beta is not None:
            stock.beta = beta
        if dividend_yield is not None:
            stock.dividend_yield = dividend_yield * 100 if dividend_yield <= 1 else dividend_yield
        if volume is not None:
            stock.avg_volume = volume
        if price is not None:
            stock.price = round(price, 2)
    except Exception as exc:
        logger.warning("Metadata refresh failed for %s: %s", stock.symbol, exc)


def _refresh_stock_history(db: Session, stock: StockProfile, ticker: Any | None = None) -> list[float]:
    if ticker is None:
        ticker = yf.Ticker(stock.symbol)

    try:
        hist = ticker.history(period="6mo", interval="1wk")
        if hist is None or hist.empty:
            raise ValueError("empty history")

        existing_rows = {
            row.date.date(): row
            for row in db.query(StockData).filter(StockData.symbol == stock.symbol).all()
        }
        history_prices: list[float] = []

        for ts, row in hist.iterrows():
            dt = ts.to_pydatetime().replace(tzinfo=None)
            existing = existing_rows.get(dt.date())
            payload = {
                "open": _safe_float(row.get("Open")),
                "high": _safe_float(row.get("High")),
                "low": _safe_float(row.get("Low")),
                "close": _safe_float(row.get("Close")),
                "volume": _safe_float(row.get("Volume")),
            }
            if existing:
                existing.open = payload["open"]
                existing.high = payload["high"]
                existing.low = payload["low"]
                existing.close = payload["close"]
                existing.volume = payload["volume"]
            else:
                db.add(StockData(symbol=stock.symbol, date=dt, **payload))
            if payload["close"] is not None:
                history_prices.append(round(payload["close"], 2))
        return history_prices
    except Exception as exc:
        logger.warning("History refresh failed for %s: %s", stock.symbol, exc)
        fallback = (
            db.query(StockData)
            .filter(StockData.symbol == stock.symbol)
            .order_by(StockData.date.asc())
            .all()
        )
        return [round(row.close, 2) for row in fallback if row.close is not None]


def _sync_stock_catalog(db: Session, stocks: list[StockProfile], include_metadata: bool = False) -> list[StockProfile]:
    ticker_map = _get_ticker_map([stock.symbol for stock in stocks])
    for stock in stocks:
        ticker = ticker_map.get(stock.symbol)
        _refresh_stock_snapshot(stock, ticker)
        if include_metadata or not stock.description or not stock.industry:
            _refresh_stock_metadata(stock, ticker)
    db.commit()
    for stock in stocks:
        db.refresh(stock)
    return stocks


def _build_dashboard_payload(stocks: list[StockProfile]) -> dict[str, Any]:
    if not stocks:
        return {
            "indices": DEFAULT_DASHBOARD_INDICES,
            "insights": [],
            "market_mood": {
                "score": 50,
                "name": "Neutral",
                "description": "Add stocks in the admin page to generate live insights.",
            },
            "sector_heatmap": [],
        }

    avg_change = sum(stock.change_pct or 0.0 for stock in stocks) / max(len(stocks), 1)
    mood_score = max(0, min(100, int(round(50 + avg_change * 8))))
    if mood_score >= 70:
        mood_name = "Bullish"
        mood_desc = "Risk appetite is elevated and gainers are leading the tape."
    elif mood_score >= 55:
        mood_name = "Moderately Bullish"
        mood_desc = "The market tone is constructive, but not overheated."
    elif mood_score <= 30:
        mood_name = "Risk-Off"
        mood_desc = "Defensive positioning is dominating the market right now."
    elif mood_score <= 45:
        mood_name = "Cautious"
        mood_desc = "Breadth is soft and traders are protecting downside."
    else:
        mood_name = "Neutral"
        mood_desc = "Signals are mixed and leadership is rotating."

    movers = sorted(stocks, key=lambda stock: stock.change_pct or 0.0, reverse=True)
    insights = []
    for stock in movers[:3] + list(reversed(movers[-2:])):
        change_pct = stock.change_pct or 0.0
        if change_pct >= 1:
            tag = "buy"
            dot = "dot-green"
            text = f"{stock.symbol} is leading today at +{change_pct:.2f}% with momentum in {stock.sector or 'its sector'}."
        elif change_pct <= -1:
            tag = "sell"
            dot = "dot-red"
            text = f"{stock.symbol} is under pressure at {change_pct:.2f}% and needs a risk check."
        else:
            tag = "hold"
            dot = "dot-amber"
            text = f"{stock.symbol} is steady at {change_pct:+.2f}% while traders wait for a clearer setup."
        insights.append({"text": text, "tag": tag, "dot": dot})

    sector_buckets: dict[str, list[float]] = {}
    for stock in stocks:
        sector = stock.sector or "Other"
        sector_buckets.setdefault(sector, []).append(stock.change_pct or 0.0)

    sector_heatmap = []
    for sector, values in sorted(sector_buckets.items(), key=lambda item: item[0]):
        average = sum(values) / max(len(values), 1)
        color = [0, 208, 156] if average >= 0 else [255, 77, 79]
        sector_heatmap.append({
            "sym": sector[:5].upper(),
            "label": sector,
            "chg": round(average, 2),
            "color": color,
        })

    return {
        "indices": DEFAULT_DASHBOARD_INDICES,
        "insights": insights[:5],
        "market_mood": {
            "score": mood_score,
            "name": mood_name,
            "description": mood_desc,
        },
        "sector_heatmap": sector_heatmap[:10],
    }


@router.post("/upload-stocks")
async def upload_stocks_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    string_io = io.StringIO(content.decode("utf-8"))
    reader = csv.DictReader(string_io)

    count = 0
    for row in reader:
        sym = row.get("symbol")
        if not sym:
            continue

        new_pred = Prediction(
            symbol=sym.strip().upper(),
            model=row.get("model", "auto"),
            predicted_price=float(row.get("price", 0)),
            direction=row.get("direction", "UP"),
            change_pct=float(row.get("change_pct", 0)),
        )
        db.add(new_pred)
        count += 1

    db.commit()
    return {"status": "success", "rows_processed": count}


@router.get("/download-stocks")
def download_stocks_csv(db: Session = Depends(get_db)):
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["symbol", "model", "predicted_price", "direction", "change_pct", "predicted_at"])

    preds = db.query(Prediction).all()
    for pred in preds:
        writer.writerow([
            pred.symbol,
            pred.model,
            pred.predicted_price,
            pred.direction,
            pred.change_pct,
            pred.predicted_at,
        ])

    response = Response(content=output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=stocks_export.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@router.get("/stocks")
def get_stocks(db: Session = Depends(get_db)):
    stocks = _get_active_stocks(db)
    stocks = _sync_stock_catalog(db, stocks)
    return [_serialize_stock_summary(stock) for stock in stocks]


@router.get("/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    stocks = _get_active_stocks(db)
    stocks = _sync_stock_catalog(db, stocks)
    return _build_dashboard_payload(stocks)


@router.get("/fundamentals/{symbol}")
def get_fundamentals(symbol: str, db: Session = Depends(get_db)):
    symbol = symbol.strip().upper()
    stock = (
        db.query(StockProfile)
        .filter(StockProfile.symbol == symbol, StockProfile.is_active.is_(True))
        .first()
    )
    if not stock:
        raise HTTPException(status_code=404, detail="Fundamentals not found for this symbol")

    ticker = None
    try:
        ticker = yf.Ticker(symbol)
    except Exception as exc:
        logger.warning("Ticker bootstrap failed for %s: %s", symbol, exc)

    _refresh_stock_snapshot(stock, ticker)
    _refresh_stock_metadata(stock, ticker)
    history = _refresh_stock_history(db, stock, ticker)
    db.commit()
    db.refresh(stock)
    return _serialize_stock_fundamentals(stock, history)


@router.get("/prices")
def get_live_prices(symbols: str, db: Session = Depends(get_db)):
    syms = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]
    if not syms:
        return {}

    stocks = (
        db.query(StockProfile)
        .filter(StockProfile.symbol.in_(syms), StockProfile.is_active.is_(True))
        .all()
    )
    _sync_stock_catalog(db, stocks)
    return {
        stock.symbol: {
            "price": round(stock.price or 0.0, 2),
            "chg": round(stock.change or 0.0, 2),
            "chgPct": round(stock.change_pct or 0.0, 2),
        }
        for stock in stocks
    }
