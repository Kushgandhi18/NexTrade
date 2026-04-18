"""
postgres.py
PostgreSQL database models and connection via SQLAlchemy async.
Tables: stocks_data, predictions, model_metrics
"""

import os
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Index, create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stock_prediction"
)

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


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


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)
    
    # Initialize demo profile if empty
    db = SessionLocal()
    try:
        if not db.query(Profile).first():
            demo_profile = Profile(name="Kush", balance=100000.0)
            db.add(demo_profile)
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
