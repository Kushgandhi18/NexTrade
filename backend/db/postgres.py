"""
postgres.py
PostgreSQL database models and connection via SQLAlchemy async.
Tables: stocks_data, predictions, model_metrics
"""

import hashlib
import os
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, String, Float, Integer, DateTime, Text, Index, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stock_prediction"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

DEFAULT_STOCK_PROFILES = [
    {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "country": "United States",
        "founded": "1976",
        "ceo": "Tim Cook",
        "price": 189.30,
        "previous_close": 187.88,
        "change": 1.42,
        "change_pct": 0.76,
    },
    {
        "symbol": "TSLA",
        "name": "Tesla Inc.",
        "sector": "Automotive",
        "industry": "Auto Manufacturers",
        "country": "United States",
        "founded": "2003",
        "ceo": "Elon Musk",
        "price": 242.80,
        "previous_close": 245.95,
        "change": -3.15,
        "change_pct": -1.28,
    },
    {
        "symbol": "MSFT",
        "name": "Microsoft Corp.",
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "country": "United States",
        "founded": "1975",
        "ceo": "Satya Nadella",
        "price": 418.50,
        "previous_close": 413.30,
        "change": 5.20,
        "change_pct": 1.26,
    },
    {
        "symbol": "GOOGL",
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "country": "United States",
        "founded": "1998",
        "ceo": "Sundar Pichai",
        "price": 174.20,
        "previous_close": 173.40,
        "change": 0.80,
        "change_pct": 0.46,
    },
    {
        "symbol": "AMZN",
        "name": "Amazon.com",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "country": "United States",
        "founded": "1994",
        "ceo": "Andy Jassy",
        "price": 186.40,
        "previous_close": 187.70,
        "change": -1.30,
        "change_pct": -0.69,
    },
    {
        "symbol": "NVDA",
        "name": "NVIDIA Corp.",
        "sector": "Technology",
        "industry": "Semiconductors",
        "country": "United States",
        "founded": "1993",
        "ceo": "Jensen Huang",
        "price": 877.39,
        "previous_close": 858.97,
        "change": 18.42,
        "change_pct": 2.14,
    },
    {
        "symbol": "META",
        "name": "Meta Platforms",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "country": "United States",
        "founded": "2004",
        "ceo": "Mark Zuckerberg",
        "price": 492.10,
        "previous_close": 484.30,
        "change": 7.80,
        "change_pct": 1.61,
    },
]

DEFAULT_INDEX_PROFILES = [
    {"symbol": "^IXIC", "name": "Nasdaq", "display_order": 1},
    {"symbol": "^DJI", "name": "Dow 30", "display_order": 2},
    {"symbol": "^GSPC", "name": "S&P 500", "display_order": 3},
    {"symbol": "GC=F", "name": "Gold", "display_order": 4},
    {"symbol": "BTC-USD", "name": "Bitcoin", "display_order": 5},
]


# ------------------------------------------------------------------
# Table Models
# ------------------------------------------------------------------

class StockData(Base):
    """Raw OHLCV data stored after each fetch."""
    __tablename__ = "stocks_data"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_stocks_data_symbol_date", "symbol", "date", unique=True),
    )


class Prediction(Base):
    """Stores each prediction + actual for tracking and drift detection."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    model = Column(String(50), nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)  # filled in next day
    last_known_price = Column(Float)
    direction = Column(String(4))       # "UP" / "DOWN"
    change_pct = Column(Float)
    predicted_at = Column(DateTime, default=datetime.utcnow)
    actual_date = Column(DateTime, nullable=True)


class ModelMetrics(Base):
    """Stores evaluation metrics after each training run."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    model = Column(String(50), nullable=False)
    version = Column(String(50))
    regime = Column(String(20))
    mae = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    direction_accuracy = Column(Float)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    train_size = Column(Integer)
    test_size = Column(Integer)
    trained_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


class StockProfile(Base):
    """Admin-managed stock catalog with cached market metadata."""
    __tablename__ = "stock_profiles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    name = Column(String(150), nullable=False)
    sector = Column(String(120), nullable=True)
    industry = Column(String(150), nullable=True)
    description = Column(Text, nullable=True)
    ceo = Column(String(120), nullable=True)
    founded = Column(String(30), nullable=True)
    country = Column(String(80), nullable=True)
    employees = Column(Integer, nullable=True)
    market_cap = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    avg_volume = Column(Float, nullable=True)
    week_52_low = Column(Float, nullable=True)
    week_52_high = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    price = Column(Float, nullable=False, default=0.0)
    previous_close = Column(Float, nullable=True)
    change = Column(Float, nullable=False, default=0.0)
    change_pct = Column(Float, nullable=False, default=0.0)
    revenue = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockFinancial(Base):
    """Quarterly or annual financial metrics for bar charts."""
    __tablename__ = "stock_financials"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    period_label = Column(String(20), nullable=False) # e.g. "Q1 2024"
    date = Column(DateTime, nullable=False)
    revenue = Column(Float, nullable=True)
    net_income = Column(Float, nullable=True)
    is_quarterly = Column(Boolean, default=True)


class StockNewsItem(Base):
    """Cached Yahoo Finance news associated with a tracked stock."""
    __tablename__ = "stock_news_items"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    external_id = Column(String(255), nullable=True, unique=True, index=True)
    title = Column(String(400), nullable=False)
    publisher = Column(String(160), nullable=True)
    summary = Column(Text, nullable=True)
    link = Column(Text, nullable=True)
    image_url = Column(Text, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_stock_news_symbol_published", "symbol", "published_at"),
    )


class MarketIndexSnapshot(Base):
    """Cached market index and macro instrument values shown on the dashboard."""
    __tablename__ = "market_index_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    name = Column(String(120), nullable=False)
    display_order = Column(Integer, default=0)
    value = Column(Float, nullable=False, default=0.0)
    previous_close = Column(Float, nullable=True)
    change = Column(Float, nullable=False, default=0.0)
    change_pct = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StockInsightSnapshot(Base):
    """Cached AI/analytics payload used by the stock detail and AI pages."""
    __tablename__ = "stock_insight_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, unique=True, index=True)
    active_model = Column(String(50), nullable=False, default="Ensemble")
    horizon = Column(String(10), nullable=False, default="1M")
    signal = Column(String(10), nullable=False, default="HOLD")
    confidence = Column(Float, nullable=False, default=50.0)
    projected_pct = Column(Float, nullable=False, default=0.0)
    target_price = Column(Float, nullable=False, default=0.0)
    ai_score = Column(Float, nullable=False, default=50.0)
    payload_json = Column(Text, nullable=False, default="{}")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdminUser(Base):
    """Simple admin user for protecting the admin console."""
    __tablename__ = "admin_users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(80), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(30), nullable=False, default="admin")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AdminSession(Base):
    """Persisted admin session token so auth remains backend-driven."""
    __tablename__ = "admin_sessions"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), nullable=False, unique=True, index=True)
    admin_user_id = Column(Integer, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_admin_sessions_user_expiry", "admin_user_id", "expires_at"),
    )


class Profile(Base):
    """User profile and balance."""
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    balance = Column(Float, default=100000.0) # $100k demo starting
    created_at = Column(DateTime, default=datetime.utcnow)


class Holding(Base):
    """Current stock positions for a user."""
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    quantity = Column(Integer, nullable=False, default=0)
    avg_price = Column(Float, nullable=False)
    user_id = Column(Integer, default=1) # Mocked for now
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_holdings_symbol_user", "symbol", "user_id", unique=True),
    )


class Transaction(Base):
    """Historical buy/sell records."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    type = Column(String(10), nullable=False) # BUY or SELL
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    total = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, default=1)

class PendingOrder(Base):
    """Pending limit, stop-loss, or take-profit orders."""
    __tablename__ = "pending_orders"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    type = Column(String(20), nullable=False) # LIMIT_BUY, STOP_LOSS
    target_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    status = Column(String(20), default="PENDING") # PENDING, FILLED, CANCELLED
    user_id = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _hash_admin_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if not db.query(Profile).first():
            demo_profile = Profile(name="Kush", balance=100000.0)
            db.add(demo_profile)
            db.commit()

        existing_symbols = {symbol for (symbol,) in db.query(StockProfile.symbol).all()}
        missing_defaults = [item for item in DEFAULT_STOCK_PROFILES if item["symbol"] not in existing_symbols]
        if missing_defaults:
            db.add_all(StockProfile(**item) for item in missing_defaults)
            db.commit()

        existing_indices = {symbol for (symbol,) in db.query(MarketIndexSnapshot.symbol).all()}
        missing_indices = [item for item in DEFAULT_INDEX_PROFILES if item["symbol"] not in existing_indices]
        if missing_indices:
            db.add_all(MarketIndexSnapshot(**item) for item in missing_indices)
            db.commit()

        admin_username = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        existing_admin = (
            db.query(AdminUser)
            .filter(AdminUser.username == admin_username)
            .first()
        )
        if not existing_admin:
            db.add(
                AdminUser(
                    username=admin_username,
                    password_hash=_hash_admin_password(admin_password),
                    role="admin",
                    is_active=True,
                )
            )
            db.commit()
    except Exception:
        pass
    finally:
        db.close()


def get_db() -> Session:
    """FastAPI dependency: yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
