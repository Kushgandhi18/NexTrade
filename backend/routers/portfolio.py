from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field, field_validator
from typing import List
from backend.db.postgres import get_db, Holding, Transaction, Profile, PendingOrder, StockProfile

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

SELL_PENDING_TYPES = {"LIMIT_SELL", "STOP_LOSS"}
BUY_PENDING_TYPES = {"LIMIT_BUY"}

# --- Schemas ---
class TradeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    type: str = Field(..., min_length=3, max_length=20)  # BUY / SELL / LIMIT_BUY / LIMIT_SELL / STOP_LOSS
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=1)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("type")
    @classmethod
    def normalize_type(cls, value: str) -> str:
        return value.strip().upper()

class HoldingResponse(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class PortfolioResponse(BaseModel):
    name: str
    balance: float
    holdings: List[HoldingResponse]


def _get_committed_balance(db: Session, user_id: int) -> float:
    pending_buys = (
        db.query(PendingOrder)
        .filter(
            PendingOrder.user_id == user_id,
            PendingOrder.type.in_(tuple(BUY_PENDING_TYPES)),
            PendingOrder.status == "PENDING",
        )
        .all()
    )
    return sum(po.quantity * po.target_price for po in pending_buys)


def _get_committed_sell_quantity(db: Session, user_id: int, symbol: str) -> int:
    pending_sells = (
        db.query(PendingOrder)
        .filter(
            PendingOrder.user_id == user_id,
            PendingOrder.symbol == symbol,
            PendingOrder.type.in_(tuple(SELL_PENDING_TYPES)),
            PendingOrder.status == "PENDING",
        )
        .all()
    )
    return sum(po.quantity for po in pending_sells)

# --- Endpoints ---

@router.get("/", response_model=PortfolioResponse)
def get_portfolio(db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.id == 1).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    holdings = db.query(Holding).filter(Holding.user_id == 1).all()
    
    return {
        "name": profile.name,
        "balance": profile.balance,
        "holdings": [
            {
                "symbol": h.symbol,
                "quantity": h.quantity,
                "avg_price": h.avg_price
            } for h in holdings if h.quantity > 0
        ]
    }

@router.post("/trade")
def execute_trade(req: TradeRequest, db: Session = Depends(get_db)):
    profile = db.query(Profile).filter(Profile.id == 1).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    stock = (
        db.query(StockProfile)
        .filter(StockProfile.symbol == req.symbol, StockProfile.is_active.is_(True))
        .first()
    )
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found in admin catalog")

    trade_type = req.type
    total_cost = req.price * req.quantity
    
    if trade_type == "BUY":
        committed_balance = _get_committed_balance(db, user_id=1)
        available_balance = profile.balance - committed_balance
        if available_balance < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance")
        
        profile.balance -= total_cost
        
        # Update holding
        holding = db.query(Holding).filter(Holding.symbol == req.symbol, Holding.user_id == 1).first()
        if holding:
            new_qty = holding.quantity + req.quantity
            new_avg = ((holding.avg_price * holding.quantity) + total_cost) / new_qty
            holding.quantity = new_qty
            holding.avg_price = new_avg
        else:
            holding = Holding(
                symbol=req.symbol,
                quantity=req.quantity,
                avg_price=req.price,
                user_id=1
            )
            db.add(holding)
            
    elif trade_type == "SELL":
        holding = db.query(Holding).filter(Holding.symbol == req.symbol, Holding.user_id == 1).first()
        committed_qty = _get_committed_sell_quantity(db, user_id=1, symbol=req.symbol)
        available_qty = holding.quantity - committed_qty if holding else 0
        if not holding or available_qty < req.quantity:
            raise HTTPException(status_code=400, detail="Insufficient quantity to sell")
        
        profile.balance += total_cost
        holding.quantity -= req.quantity
        # Note: avg_price usually doesn't change on sell in many accounting models (FIFO/Avg Cost)

    elif trade_type in ["LIMIT_BUY", "LIMIT_SELL", "STOP_LOSS"]:
        if trade_type == "LIMIT_BUY":
            committed_balance = _get_committed_balance(db, user_id=1)
            if profile.balance - committed_balance < total_cost:
                raise HTTPException(status_code=400, detail="Insufficient balance for limit order")
        else:
            holding = db.query(Holding).filter(Holding.symbol == req.symbol, Holding.user_id == 1).first()
            committed_qty = _get_committed_sell_quantity(db, user_id=1, symbol=req.symbol)
            if not holding or (holding.quantity - committed_qty) < req.quantity:
                raise HTTPException(status_code=400, detail="Insufficient quantity for pending sell order")
                
        pending = PendingOrder(
            symbol=req.symbol,
            type=trade_type,
            target_price=req.price,
            quantity=req.quantity,
            user_id=1
        )
        db.add(pending)
        db.commit()
        return {"status": "success", "msg": f"Order {trade_type} placed successfully.", "new_balance": profile.balance}
        
    else:
        raise HTTPException(status_code=400, detail="Invalid trade type")

    # Record transaction
    txn = Transaction(
        symbol=req.symbol,
        type=trade_type,
        price=req.price,
        quantity=req.quantity,
        total=total_cost,
        user_id=1
    )
    db.add(txn)
    
    db.commit()
    return {"status": "success", "new_balance": profile.balance}

@router.get("/transactions")
def get_transactions(db: Session = Depends(get_db)):
    txns = (
        db.query(Transaction)
        .filter(Transaction.user_id == 1)
        .order_by(Transaction.timestamp.desc())
        .all()
    )
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "side": t.type,
            "price": t.price,
            "quantity": t.quantity,
            "total": t.total,
            "timestamp": t.timestamp
        } for t in txns
    ]

@router.get("/pending")
def get_pending_orders(db: Session = Depends(get_db)):
    orders = db.query(PendingOrder).filter(PendingOrder.user_id == 1, PendingOrder.status == "PENDING").all()
    return [
        {
            "id": o.id,
            "symbol": o.symbol,
            "type": o.type,
            "target": o.target_price,
            "quantity": o.quantity,
            "created_at": o.created_at
        } for o in orders
    ]
