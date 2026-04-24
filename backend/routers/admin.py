from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from backend.db.postgres import StockProfile, get_db

router = APIRouter(prefix="/admin", tags=["Admin"])


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


@router.get("/stocks")
def list_admin_stocks(db: Session = Depends(get_db)):
    stocks = db.query(StockProfile).order_by(StockProfile.symbol.asc()).all()
    return [_serialize_admin_stock(stock) for stock in stocks]


@router.get("/stocks/{symbol}")
def get_admin_stock(symbol: str, db: Session = Depends(get_db)):
    stock = db.query(StockProfile).filter(StockProfile.symbol == symbol.strip().upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")
    return _serialize_admin_stock(stock)


@router.post("/stocks")
def create_stock(payload: StockPayload, db: Session = Depends(get_db)):
    existing = db.query(StockProfile).filter(StockProfile.symbol == payload.symbol).first()
    if existing:
        raise HTTPException(status_code=409, detail="Stock already exists")

    stock = StockProfile(symbol=payload.symbol)
    _apply_payload(stock, payload)
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return _serialize_admin_stock(stock)


@router.put("/stocks/{symbol}")
def update_stock(symbol: str, payload: StockUpdatePayload, db: Session = Depends(get_db)):
    stock = db.query(StockProfile).filter(StockProfile.symbol == symbol.strip().upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    _apply_payload(stock, payload)
    db.commit()
    db.refresh(stock)
    return _serialize_admin_stock(stock)
