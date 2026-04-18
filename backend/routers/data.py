import csv
import io
from fastapi import APIRouter, Depends, UploadFile, File, Response
from sqlalchemy.orm import Session
from backend.db.postgres import get_db, StockData, Prediction, ModelMetrics

router = APIRouter(prefix="/data", tags=["Data IO"])

@router.post("/upload-stocks")
async def upload_stocks_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    string_io = io.StringIO(content.decode("utf-8"))
    reader = csv.DictReader(string_io)
    
    count = 0
    for row in reader:
        # Simple bulk insert/update into predictions table for demo purposes
        # In real scenario, would map to specific tables
        sym = row.get("symbol")
        if not sym: continue
        
        # Upsert logic or just insert new ones
        new_pred = Prediction(
            symbol=sym,
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
    
    # Header
    writer.writerow(["symbol", "model", "predicted_price", "direction", "change_pct", "predicted_at"])
    
    preds = db.query(Prediction).all()
    for p in preds:
        writer.writerow([p.symbol, p.model, p.predicted_price, p.direction, p.change_pct, p.predicted_at])
    
    response = Response(content=output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=stocks_export.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
