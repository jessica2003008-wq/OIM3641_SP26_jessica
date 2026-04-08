from fastapi import FastAPI
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.get("/analyze/{symbol}")
async def analyze_stock(symbol: str):
    response = (
        supabase.table("stock_records")
        .select("*")
        .eq("ticker", symbol.upper())
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not response.data:
        return {"error": "No data found in database. Run ingest.py first."}

    record = response.data[0]
    initial_p = record["initial_price"]
    final_p = record["final_price"]

    signal = "Bullish" if final_p > initial_p else "Bearish"

    return {
        "ticker": symbol.upper(),
        "analysis": {
            "start_price": initial_p,
            "current_price": final_p,
            "signal": signal,
            "source": "Internal Database"
        }
    }