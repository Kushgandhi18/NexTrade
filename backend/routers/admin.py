import logging
import os
import secrets
from datetime import datetime, timedelta
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.db.postgres import (
    AdminSession,
    AdminUser,
    SessionLocal,
    StockProfile,
    _hash_admin_password,
    get_db,
)
from backend.services.sync_service import full_stock_sync

router = APIRouter(prefix="/admin", tags=["Admin"])
SESSION_TTL_HOURS = max(1, int(os.environ.get("ADMIN_SESSION_HOURS", "12")))

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

def _bg_sync_task(symbol: str):
    db = SessionLocal()
    try:
        full_stock_sync(db, symbol)
    finally:
        db.close()


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



def _build_yahoo_payload(symbol: str, is_active: bool = True) -> dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        info = getattr(ticker, "info", None) or {}
        fast_info = getattr(ticker, "fast_info", None)

        price = (
            _safe_float(_fast_info_value(fast_info, "last_price"))
            or _safe_float(info.get("currentPrice"))
            or _safe_float(info.get("regularMarketPrice"))
            or 0.0
        )
        previous_close = (
            _safe_float(_fast_info_value(fast_info, "previous_close"))
            or _safe_float(info.get("previousClose"))
        )
        volume = (
            _safe_float(_fast_info_value(fast_info, "last_volume"))
            or _safe_float(_fast_info_value(fast_info, "ten_day_average_volume"))
            or _safe_float(info.get("volume"))
        )
        dividend_yield = _safe_float(info.get("dividendYield"))
        if dividend_yield is not None and dividend_yield <= 1:
            dividend_yield *= 100

        name = info.get("shortName") or info.get("longName") or symbol
        if name == symbol and price == 0:
            raise ValueError("Could not fetch stock data from Yahoo Finance")

        return {
            "symbol": symbol,
            "name": name,
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "description": info.get("longBusinessSummary"),
            "ceo": _extract_ceo_name(info),
            "founded": KNOWN_FOUNDED.get(symbol) or str(info.get("ipoExpectedDate") or "") or None,
            "country": info.get("country"),
            "employees": _safe_int(info.get("fullTimeEmployees")),
            "market_cap": _safe_float(info.get("marketCap")),
            "pe_ratio": _safe_float(info.get("trailingPE")),
            "eps": _safe_float(info.get("trailingEps")),
            "avg_volume": volume,
            "week_52_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "week_52_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "beta": _safe_float(info.get("beta")),
            "dividend_yield": dividend_yield,
            "price": round(price, 2),
            "previous_close": round(previous_close, 2) if previous_close is not None else None,
            "is_active": is_active,
        }
    except Exception as exc:
        import random
        # Try one more time with a different User-Agent if it's a rate limit
        if "Too Many Requests" in str(exc) or "Rate limit" in str(exc):
            logger.warning(f"Yahoo Finance rate limited {symbol}. Retrying once...")
            try:
                # Add a tiny jitter
                import time
                time.sleep(random.uniform(0.5, 1.5))
                ticker = yf.Ticker(symbol)
                # Force a fresh fetch
                info = ticker.info
                # ... repeat extraction logic or just use a more stable fallback
                if info and info.get("currentPrice"):
                     # If we got it on retry, great!
                     return _build_yahoo_payload(symbol, is_active)
            except:
                pass
        
        logger.error(f"Failed to fetch real data for {symbol}: {exc}")
        # Return what we have or a more obvious error state than $100.00
        return {
            "symbol": symbol,
            "name": symbol,
            "price": 0.0,
            "previous_close": 0.0,
            "is_active": is_active,
            "description": f"Sync Pending: Yahoo Finance is currently rate-limiting this server. Real data will appear shortly.",
        }
        logger.error(f"Unexpected error fetching data for {symbol}: {exc}")
        raise



def _parse_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()



def _serialize_admin_user(admin_user: AdminUser) -> dict[str, Any]:
    return {
        "id": admin_user.id,
        "username": admin_user.username,
        "role": admin_user.role,
    }



def require_admin(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    token = _parse_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin authentication required")

    session = db.query(AdminSession).filter(AdminSession.token == token).first()
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin session is invalid")

    if session.expires_at <= datetime.utcnow():
        db.delete(session)
        db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Admin session expired")

    admin_user = (
        db.query(AdminUser)
        .filter(AdminUser.id == session.admin_user_id, AdminUser.is_active.is_(True))
        .first()
    )
    if not admin_user or admin_user.role.lower() != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access denied")

    return {"user": admin_user, "session": session}



def _normalize_search_results(results: list[dict[str, Any]], tracked_symbols: set[str]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in results:
        symbol = str(item.get("symbol") or item.get("displaySymbol") or "").strip().upper()
        name = item.get("shortname") or item.get("longname") or item.get("displayName") or symbol
        quote_type = str(item.get("quoteType") or item.get("typeDisp") or "").upper()
        exchange = item.get("exchange") or item.get("exchangeDisp") or item.get("fullExchangeName")
        region = item.get("region") or item.get("market")

        if not symbol or symbol in seen:
            continue
        if quote_type and quote_type not in {"EQUITY", "ETF", "MUTUALFUND", "INDEX", "CRYPTOCURRENCY"}:
            continue

        seen.add(symbol)
        normalized.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "region": region,
                "quote_type": quote_type or "UNKNOWN",
                "already_tracked": symbol in tracked_symbols,
            }
        )

    return normalized[:8]


class AdminLoginPayload(BaseModel):
    username: str = Field(..., min_length=1, max_length=80)
    password: str = Field(..., min_length=1, max_length=255)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        return value.strip()


class StockPayload(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    name: str = Field(..., min_length=1, max_length=150)
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    ceo: str | None = None
    founded: str | None = None
    country: str | None = None
    employees: int | None = Field(default=None, ge=0)
    market_cap: float | None = Field(default=None, ge=0)
    pe_ratio: float | None = None
    eps: float | None = None
    avg_volume: float | None = Field(default=None, ge=0)
    week_52_low: float | None = Field(default=None, ge=0)
    week_52_high: float | None = Field(default=None, ge=0)
    beta: float | None = None
    dividend_yield: float | None = Field(default=None, ge=0)
    price: float = Field(default=0.0, ge=0)
    previous_close: float | None = Field(default=None, ge=0)
    is_active: bool = True

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("name", "sector", "industry", "description", "ceo", "founded", "country", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class StockUpdatePayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    ceo: str | None = None
    founded: str | None = None
    country: str | None = None
    employees: int | None = Field(default=None, ge=0)
    market_cap: float | None = Field(default=None, ge=0)
    pe_ratio: float | None = None
    eps: float | None = None
    avg_volume: float | None = Field(default=None, ge=0)
    week_52_low: float | None = Field(default=None, ge=0)
    week_52_high: float | None = Field(default=None, ge=0)
    beta: float | None = None
    dividend_yield: float | None = Field(default=None, ge=0)
    price: float = Field(default=0.0, ge=0)
    previous_close: float | None = Field(default=None, ge=0)
    is_active: bool = True

    @field_validator("name", "sector", "industry", "description", "ceo", "founded", "country", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class StockLookupPayload(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    is_active: bool = True

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()



def _serialize_admin_stock(stock: StockProfile) -> dict[str, Any]:
    return {
        "id": stock.id,
        "symbol": stock.symbol,
        "name": stock.name,
        "sector": stock.sector,
        "industry": stock.industry,
        "description": stock.description,
        "ceo": stock.ceo,
        "founded": stock.founded,
        "country": stock.country,
        "employees": stock.employees,
        "market_cap": stock.market_cap,
        "pe_ratio": stock.pe_ratio,
        "eps": stock.eps,
        "avg_volume": stock.avg_volume,
        "week_52_low": stock.week_52_low,
        "week_52_high": stock.week_52_high,
        "beta": stock.beta,
        "dividend_yield": stock.dividend_yield,
        "price": stock.price,
        "previous_close": stock.previous_close,
        "change": stock.change,
        "change_pct": stock.change_pct,
        "is_active": bool(stock.is_active),
        "updated_at": stock.updated_at.isoformat() if stock.updated_at else None,
    }



def _apply_payload(stock: StockProfile, payload: StockPayload | StockUpdatePayload) -> None:
    stock.name = payload.name
    stock.sector = payload.sector
    stock.industry = payload.industry
    stock.description = payload.description
    stock.ceo = payload.ceo
    stock.founded = payload.founded
    stock.country = payload.country
    stock.employees = payload.employees
    stock.market_cap = payload.market_cap
    stock.pe_ratio = payload.pe_ratio
    stock.eps = payload.eps
    stock.avg_volume = payload.avg_volume
    stock.week_52_low = payload.week_52_low
    stock.week_52_high = payload.week_52_high
    stock.beta = payload.beta
    stock.dividend_yield = payload.dividend_yield
    stock.price = payload.price
    stock.previous_close = payload.previous_close
    if payload.previous_close not in (None, 0):
        stock.change = round(payload.price - payload.previous_close, 2)
        stock.change_pct = round(((payload.price - payload.previous_close) / payload.previous_close) * 100, 2)
    else:
        stock.change = 0.0
        stock.change_pct = 0.0
    stock.is_active = payload.is_active


@router.post("/auth/login")
def admin_login(payload: AdminLoginPayload, db: Session = Depends(get_db)):
    admin_user = (
        db.query(AdminUser)
        .filter(AdminUser.username == payload.username, AdminUser.is_active.is_(True))
        .first()
    )
    if not admin_user or admin_user.password_hash != _hash_admin_password(payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin credentials")

    db.query(AdminSession).filter(AdminSession.expires_at <= datetime.utcnow()).delete()
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=SESSION_TTL_HOURS)
    session = AdminSession(token=token, admin_user_id=admin_user.id, expires_at=expires_at)
    db.add(session)
    db.commit()

    return {
        "status": "success",
        "token": token,
        "expires_at": expires_at.isoformat(),
        "user": _serialize_admin_user(admin_user),
    }


@router.get("/auth/status")
def admin_status(admin: dict[str, Any] = Depends(require_admin)):
    return {
        "authenticated": True,
        "user": _serialize_admin_user(admin["user"]),
        "expires_at": admin["session"].expires_at.isoformat(),
    }


@router.post("/auth/logout")
def admin_logout(
    admin: dict[str, Any] = Depends(require_admin),
    db: Session = Depends(get_db),
):
    db.delete(admin["session"])
    db.commit()
    return {"status": "success"}


@router.get("/stocks/search")
def search_admin_stocks(
    q: str = Query(..., min_length=1, max_length=120),
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Search query is required")

    tracked_symbols = {symbol for (symbol,) in db.query(StockProfile.symbol).all()}

    # Try yf.Search first (yfinance >= 0.2.52), fall back to direct Yahoo HTTP API
    raw_quotes: list[dict[str, Any]] = []
    try:
        import yfinance as yf
        search_cls = getattr(yf, "Search", None)
        if search_cls is not None:
            search = search_cls(
                query,
                max_results=8,
                news_count=0,
                lists_count=0,
                include_cb=False,
                include_nav_links=False,
                enable_fuzzy_query=True,
                raise_errors=False,
            )
            raw_quotes = getattr(search, "quotes", None) or []
    except Exception:
        pass

    if not raw_quotes:
        # Fallback: direct Yahoo Finance search API
        try:
            import httpx
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            headers = {"User-Agent": "Mozilla/5.0"}
            params = {
                "q": query,
                "quotesCount": 8,
                "newsCount": 0,
                "enableFuzzyQuery": True,
                "quotesQueryId": "tss_match_phrase_query",
            }
            resp = httpx.get(url, params=params, headers=headers, timeout=6.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_quotes = data.get("quotes", []) or []
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Yahoo Finance search failed: {exc}") from exc

    return _normalize_search_results(raw_quotes, tracked_symbols)



@router.get("/stocks")
def list_admin_stocks(
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    stocks = db.query(StockProfile).order_by(StockProfile.symbol.asc()).all()
    return [_serialize_admin_stock(stock) for stock in stocks]


@router.get("/stocks/{symbol}")
def get_admin_stock(
    symbol: str,
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    stock = db.query(StockProfile).filter(StockProfile.symbol == symbol.strip().upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    return _serialize_admin_stock(stock)


@router.post("/stocks/fetch")
def fetch_stock_from_yahoo(
    payload: StockLookupPayload,
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    return _build_yahoo_payload(payload.symbol, payload.is_active)


@router.post("/stocks")
def create_stock(
    payload: StockPayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    existing = db.query(StockProfile).filter(StockProfile.symbol == payload.symbol).first()
    if existing:
        raise HTTPException(status_code=409, detail="Stock already exists")

    stock = StockProfile(symbol=payload.symbol)
    _apply_payload(stock, payload)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    
    # Trigger deep sync immediately in background
    background_tasks.add_task(_bg_sync_task, stock.symbol)
    
    return _serialize_admin_stock(stock)


@router.put("/stocks/{symbol}")
def update_stock(
    symbol: str,
    payload: StockUpdatePayload,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    stock = db.query(StockProfile).filter(StockProfile.symbol == symbol.strip().upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    _apply_payload(stock, payload)
    db.commit()
    db.refresh(stock)
    
    # Trigger deep sync immediately in background
    background_tasks.add_task(_bg_sync_task, stock.symbol)
    
    return _serialize_admin_stock(stock)


@router.post("/stocks/{symbol}/refresh")
def refresh_stock_from_yahoo(
    symbol: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    admin: dict[str, Any] = Depends(require_admin),
):
    del admin
    normalized_symbol = symbol.strip().upper()
    stock = db.query(StockProfile).filter(StockProfile.symbol == normalized_symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    # This handles the metadata and basic snapshot
    yahoo_payload = StockPayload(**_build_yahoo_payload(normalized_symbol, bool(stock.is_active)))
    _apply_payload(stock, yahoo_payload)
    db.commit()
    
    # Trigger deep sync (History, News) immediately in background
    background_tasks.add_task(_bg_sync_task, normalized_symbol)
    
    db.refresh(stock)
    return _serialize_admin_stock(stock)
