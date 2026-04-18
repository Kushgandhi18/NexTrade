from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from backend.db.postgres import get_db, Holding, Transaction, Profile

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

# --- Schemas ---
class TradeRequest(BaseModel):
    symbol: str
    type: str # BUY / SELL
    price: float
    quantity: int

class HoldingResponse(BaseModel):
    symbol: str
    quantity: int
    avg_price: float

class PortfolioResponse(BaseModel):
    name: str
    balance: float
    holdings: List[HoldingResponse]

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

    total_cost = req.price * req.quantity
    
    if req.type.upper() == "BUY":
        if profile.balance < total_cost:
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
            
    elif req.type.upper() == "SELL":
        holding = db.query(Holding).filter(Holding.symbol == req.symbol, Holding.user_id == 1).first()
        if not holding or holding.quantity < req.quantity:
            raise HTTPException(status_code=400, detail="Insufficient quantity to sell")
        
        profile.balance += total_cost
        holding.quantity -= req.quantity
        # Note: avg_price usually doesn't change on sell in many accounting models (FIFO/Avg Cost)
        
    else:
        raise HTTPException(status_code=400, detail="Invalid trade type")

    # Record transaction
    txn = Transaction(
        symbol=req.symbol,
        type=req.type.upper(),
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
    txns = db.query(Transaction).filter(Transaction.user_id == 1).all()
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
