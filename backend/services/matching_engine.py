import logging
import asyncio
import yfinance as yf
from sqlalchemy.orm import Session
from backend.db.postgres import SessionLocal, PendingOrder, Profile, Holding, Transaction

logger = logging.getLogger(__name__)

class OrderMatchingEngine:
    """Runs a periodic check against live prices to fill Limit/SL orders."""

    @classmethod
    async def run_loop(cls):
        while True:
            try:
                # Use asyncio.to_thread for blocking DB queries
                await asyncio.to_thread(cls._match_orders_sync)
            except asyncio.CancelledError:
                logger.info("Order Matching Engine shutting down.")
                break
            except Exception as e:
                logger.error(f"Error in matching engine: {e}")
            await asyncio.sleep(5)  # Check every 5 seconds

    @classmethod
    def _match_orders_sync(cls):
        db: Session = SessionLocal()
        try:
            # Prevent race conditions with skip_locked
            orders = db.query(PendingOrder).with_for_update(skip_locked=True).filter(PendingOrder.status == "PENDING").all()
            if not orders:
                return

            symbols = list(set(o.symbol for o in orders))
            # Batch fetch prices using yfinance
            tickers = yf.Tickers(" ".join(symbols))
            live_prices = {}
            for sym in symbols:
                try:
                    price = tickers.tickers[sym].fast_info.last_price
                    live_prices[sym] = price
                except Exception as e:
                    logger.warning(f"Failed to fetch price for {sym}: {e}")

            for order in orders:
                live_price = live_prices.get(order.symbol)
                if not live_price:
                    continue

                execute = False
                if order.type == "LIMIT_BUY" and live_price <= order.target_price:
                    execute = True
                elif order.type == "LIMIT_SELL" and live_price >= order.target_price:
                    execute = True
                elif order.type == "STOP_LOSS" and live_price <= order.target_price:
                    execute = True

                if execute:
                    # Execute the order!
                    profile = db.query(Profile).with_for_update().filter(Profile.id == order.user_id).first()
                    if not profile:
                        logger.error(f"Cannot execute order {order.id}: Profile {order.user_id} not found.")
                        order.status = "CANCELLED"
                        continue

                    total_cost = live_price * order.quantity
                    
                    if order.type == "LIMIT_BUY":
                        if profile.balance >= total_cost:
                            profile.balance -= total_cost
                            holding = db.query(Holding).filter(Holding.symbol == order.symbol, Holding.user_id == order.user_id).first()
                            if holding:
                                new_qty = holding.quantity + order.quantity
                                new_avg = ((holding.avg_price * holding.quantity) + total_cost) / new_qty
                                holding.quantity = new_qty
                                holding.avg_price = new_avg
                            else:
                                holding = Holding(symbol=order.symbol, quantity=order.quantity, avg_price=live_price, user_id=order.user_id)
                                db.add(holding)
                            
                            txn = Transaction(symbol=order.symbol, type="BUY", price=live_price, quantity=order.quantity, total=total_cost, user_id=order.user_id)
                            db.add(txn)
                            order.status = "FILLED"
                        else:
                            # Insufficient funds to fill
                            order.status = "CANCELLED"
                            logger.warning(f"LIMIT_BUY {order.id} cancelled due to insufficient balance.")
                    
                    elif order.type in {"LIMIT_SELL", "STOP_LOSS"}:
                        holding = db.query(Holding).filter(Holding.symbol == order.symbol, Holding.user_id == order.user_id).first()
                        if holding and holding.quantity >= order.quantity:
                            profile.balance += total_cost
                            holding.quantity -= order.quantity
                            
                            txn = Transaction(symbol=order.symbol, type="SELL", price=live_price, quantity=order.quantity, total=total_cost, user_id=order.user_id)
                            db.add(txn)
                            order.status = "FILLED"
                        else:
                            order.status = "CANCELLED"
                            
            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
