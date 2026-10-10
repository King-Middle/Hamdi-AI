"""
MetaTraderBot.py
================
Live trading bot using MetaAPI Cloud SDK (works on Ubuntu/Linux).
Replicates the strategy from Simulation 2.0.py.

STRATEGY SUMMARY:
  - Uses a trained Keras LSTM model + macro market data to predict the next-day % change
    for 56 US stocks.
  - Takes LONG  positions when prediction > 0 AND yesterday's close change was negative.
  - Takes SHORT positions when prediction < 0 AND yesterday's close change was positive.
  - TP = entry +/- |prediction|%  ;  SL = entry -/+ |prediction| x RISK_PROFIT_RATIO %
  - Each position uses 1/10 of available margin (max 10 simultaneous positions).
  - MetaAPI natively manages TP/SL execution on the broker side.
  - Positions open for >= 10 trading days are force-closed at the start of each daily cycle.
  - The bot runs ONE cycle per trading day (at 09:35 ET) then sleeps until the next day.

REQUIREMENTS (Ubuntu / any Linux):
  pip install metaapi-cloud-sdk tensorflow scikit-learn yfinance pandas numpy pytz asyncio

HOW MetaAPI WORKS:
  1. You sign up at https://app.metaapi.cloud (free tier available).
  2. You connect your MT5 broker account in the MetaAPI dashboard.
  3. MetaAPI gives you an API_TOKEN and an ACCOUNT_ID.
  4. This script uses those to trade via the cloud - no Windows needed.

IMPORTANT: Fill in all fields in the USER CONFIGURATION section below before running.
"""

# ============================================================
#                   USER CONFIGURATION
# ============================================================

# -- MetaAPI credentials --
METAAPI_TOKEN   = "YOUR_METAAPI_TOKEN"
# Get this from: https://app.metaapi.cloud/token  (after signing up)

METAAPI_ACCOUNT_ID = "YOUR_ACCOUNT_ID"
# Get this from: https://app.metaapi.cloud/accounts
# (create an account there and link it to your MT5 broker)

# -- Model --
MODEL_PATH = "/root/Hamdi/Hamdi_App copy/Hamdi_SeriUretim_15_2.0.keras"
# Full path to the .keras model file on your Ubuntu server.
# Update the exact filename if needed (check with: ls "/root/Hamdi/Hamdi_App copy/")

# -- Strategy Parameters (must match Simulation 2.0.py) --
MAX_POSITIONS         = 10      # Maximum simultaneous open positions
RISK_PROFIT_RATIO     = 1       # SL distance = TP distance x this ratio
MIN_PREDICTION_PCT    = 0.0     # Minimum |prediction|% required to trade (0 = no filter)
UNDER_3_PCT_TP_FACTOR = 1       # TP multiplier when |prediction| < 3%
OPPOSITE_DIR_REQUIRED = True    # Only trade opposite to yesterday's price direction
MAX_TRADE_DAYS        = 10      # Force-close positions held this many trading days
TRADE_HOUR_ET         = 9       # Hour   (US Eastern Time) to run the daily cycle
TRADE_MINUTE_ET       = 35      # Minute (US Eastern Time) to run the daily cycle

# ============================================================
#  Symbol suffix map: your broker uses .nys (NYSE) or .nas (NASDAQ)
#  Bot will try the mapped suffix first, then fallback to the other.
# ============================================================
SYMBOL_SUFFIX_MAP = {
    # ---- NYSE stocks (.nys) ----
    'AES':  '.nys', 'AIG':  '.nys', 'APA':  '.nys', 'ATO':  '.nys',
    'AWK':  '.nys', 'BAC':  '.nys', 'BEN':  '.nys', 'CF':   '.nys',
    'CI':   '.nys', 'CNC':  '.nys', 'CNP':  '.nys', 'COP':  '.nys',
    'CVS':  '.nys', 'CVX':  '.nys', 'D':    '.nys', 'DUK':  '.nys',
    'ED':   '.nys', 'EQR':  '.nys', 'FE':   '.nys', 'GIS':  '.nys',
    'HAL':  '.nys', 'HPE':  '.nys', 'HBAN': '.nys', 'IVZ':  '.nys',
    'J':    '.nys', 'JNJ':  '.nys', 'KHC':  '.nys', 'KIM':  '.nys',
    'LHX':  '.nys', 'LMT':  '.nys', 'MO':   '.nys', 'NWS':  '.nys',
    'NWSA': '.nys', 'OXY':  '.nys', 'PPL':  '.nys', 'PSX':  '.nys',
    'SCHW': '.nys', 'SO':   '.nys', 'STZ':  '.nys', 'T':    '.nys',
    'TSN':  '.nys', 'WFC':  '.nys', 'WY':   '.nys', 'XOM':  '.nys',
    # ---- NASDAQ stocks (.nas) ----
    'AMGN': '.nas', 'BIIB': '.nas', 'CME':  '.nas', 'FFIV': '.nas',
    'FOXA': '.nas', 'FOX':  '.nas', 'NDAQ': '.nas', 'ORLY': '.nas',
    'TSCO': '.nas', 'VRTX': '.nas', 'XRAY': '.nas',
}

# ============================================================

import os
import sys
import time
import asyncio
import logging
import traceback
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging - writes to both console and bot_log.txt
# ---------------------------------------------------------------------------
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heavy imports
# ---------------------------------------------------------------------------
try:
    from metaapi_cloud_sdk import MetaApi
except ImportError:
    log.error("metaapi-cloud-sdk not found. Run: pip install metaapi-cloud-sdk")
    sys.exit(1)

try:
    from tensorflow.keras.models import load_model  # type: ignore
except ImportError:
    log.error("TensorFlow not found. Run: pip install tensorflow")
    sys.exit(1)

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    log.error("scikit-learn not found. Run: pip install scikit-learn")
    sys.exit(1)

try:
    from yfinance import download as yf_download
except ImportError:
    log.error("yfinance not found. Run: pip install yfinance")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Symbol list (same as Simulation 2.0.py)
# ---------------------------------------------------------------------------
SYMBOL_LIST = [
    'AES', 'AIG', 'AMGN', 'APA', 'ATO', 'AWK', 'BAC', 'BIIB', 'CNP', 'CVX',
    'DUK', 'FOXA', 'MO', 'NDAQ', 'T', 'CNC', 'CF', 'SCHW', 'CI', 'CME',
    'COP', 'ED', 'STZ', 'CVS', 'XRAY', 'D', 'EQR', 'XOM', 'FFIV', 'FE',
    'FOX', 'BEN', 'GIS', 'HAL', 'HPE', 'HBAN', 'IVZ', 'J', 'JNJ', 'KIM',
    'KHC', 'LHX', 'LMT', 'NWSA', 'NWS', 'ORLY', 'OXY', 'PSX', 'PPL', 'SO',
    'TSCO', 'TSN', 'VRTX', 'WFC', 'WY',
]

# ---------------------------------------------------------------------------
# Macro symbols used as LSTM features (same as Simulation 2.0.py)
# ---------------------------------------------------------------------------
MACRO_SYMBOLS = {
    'VIX':         '^VIX',
    'US10Y':       '^TNX',
    'US3M':        '^IRX',
    'US30Y':       '^TYX',
    'XAUUSD':      'GC=F',
    'XAGUSD':      'SI=F',
    'WTI':         'USO',
    'Brent':       'BZ=F',
    'NaturalGas':  'NG=F',
    'Copper':      'HG=F',
    'Platinum':    'PL=F',
    'Palladium':   'PA=F',
    'SPX':         '^GSPC',
    'DJI':         '^DJI',
    'IXIC':        '^IXIC',
    'RUT':         '^RUT',
    'FTSE':        '^FTSE',
    'Nikkei':      '^N225',
    'HangSeng':    '^HSI',
    'EuroStoxx50': 'FEZ',
    'BTC':         'BTC-USD',
    'ETH':         'ETH-USD',
    'SOL':         'SOL-USD',
}

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_model  = None
_scaler = None

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model_and_scaler():
    """Load the Keras model once at startup. Scaler is a plain MinMaxScaler
    fitted fresh on every prediction call (identical to the simulation)."""
    global _model, _scaler
    log.info(f"Loading model from: {MODEL_PATH}")
    _model  = load_model(MODEL_PATH)
    _scaler = MinMaxScaler()
    log.info("Model loaded. MinMaxScaler initialised.")

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def get_stock_and_macro_data(symbol: str, start_date: date, end_date: date):
    """Download stock OHLCV + all macro indicator data. Returns a DataFrame or None."""
    try:
        stock_df = yf_download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d', auto_adjust=True, progress=False, timeout=30
        )
        if isinstance(stock_df.columns, pd.MultiIndex):
            stock_df.columns = stock_df.columns.droplevel('Ticker')
        if stock_df.empty:
            log.warning(f"No yfinance data for {symbol}")
            return None

        for name, ticker in MACRO_SYMBOLS.items():
            try:
                temp = yf_download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d', auto_adjust=True, progress=False, timeout=30
                )
                if isinstance(temp.columns, pd.MultiIndex):
                    temp.columns = temp.columns.droplevel('Ticker')
                temp = temp[['Close']].rename(columns={'Close': name})
                stock_df = stock_df.join(temp, how='inner')
            except Exception as e:
                log.warning(f"Macro fetch failed for {name}: {e}")

        stock_df.dropna(inplace=True)
        return stock_df

    except Exception as e:
        log.error(f"get_stock_and_macro_data({symbol}): {e}")
        return None


def predict_symbol(symbol: str):
    """Run LSTM model on `symbol`. Returns predicted % change, or None on failure."""
    global _model, _scaler

    today      = date.today()
    start_data = today - timedelta(days=180)

    df = get_stock_and_macro_data(symbol, start_data, today + timedelta(days=1))
    if df is None or len(df) < 5:
        return None

    features = ['Open', 'High', 'Low', 'Close', 'Volume'] + list(MACRO_SYMBOLS.keys())
    features = [f for f in features if f in df.columns]
    data     = df[features].copy()

    try:
        scaled  = _scaler.fit_transform(data)
        X_input = scaled.reshape(1, scaled.shape[0], scaled.shape[1])
        return float(_model.predict(X_input, verbose=0)[0][0])
    except Exception as e:
        log.error(f"predict_symbol({symbol}): {e}")
        return None


def get_yesterday_change(symbol: str):
    """Return close-to-close change between last two trading days. Positive = up."""
    try:
        end   = date.today() + timedelta(days=1)
        start = date.today() - timedelta(days=10)
        df = yf_download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval='1d', auto_adjust=True, progress=False, timeout=15
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel('Ticker')
        closes = df['Close'].dropna().tolist()
        if len(closes) < 2:
            return None
        return closes[-1] - closes[-2]
    except Exception as e:
        log.warning(f"get_yesterday_change({symbol}): {e}")
        return None

# ---------------------------------------------------------------------------
# Symbol suffix helpers
# ---------------------------------------------------------------------------

def broker_symbol(raw: str) -> str:
    """Return the broker-formatted symbol name (e.g. 'AES.nys')."""
    suffix = SYMBOL_SUFFIX_MAP.get(raw, '.nys')   # default to .nys if unknown
    return raw + suffix


def base_symbol(broker_sym: str) -> str:
    """Strip .nys / .nas suffix to get the plain ticker."""
    for suffix in ('.nys', '.nas'):
        if broker_sym.lower().endswith(suffix):
            return broker_sym[: -len(suffix)]
    return broker_sym

# ---------------------------------------------------------------------------
# Trading day counter
# ---------------------------------------------------------------------------

def count_trading_days_open(open_time_iso: str) -> int:
    """
    Count weekdays elapsed since `open_time_iso` (ISO-8601 string from MetaAPI).
    Ignores market holidays for simplicity.
    """
    try:
        open_dt = datetime.fromisoformat(open_time_iso.replace('Z', '+00:00')).date()
    except Exception:
        return 0
    today   = date.today()
    count   = 0
    current = open_dt
    while current < today:
        if current.weekday() < 5:   # Mon=0 ... Fri=4
            count += 1
        current += timedelta(days=1)
    return count

# ---------------------------------------------------------------------------
# Lot size calculation
# ---------------------------------------------------------------------------

def calculate_lot_size(price: float, contract_size: float,
                        min_lot: float, max_lot: float,
                        lot_step: float, budget_usd: float) -> float:
    """
    Calculate lot size such that position value ~= budget_usd.
    lot = budget / (contract_size * price), rounded down to lot_step.
    """
    if price <= 0 or contract_size <= 0 or lot_step <= 0:
        return 0.0
    raw_lot = budget_usd / (contract_size * price)
    lot     = int(raw_lot / lot_step) * lot_step
    lot     = max(min_lot, min(max_lot, lot))
    return round(lot, 8)

# ---------------------------------------------------------------------------
# Main async trading logic
# ---------------------------------------------------------------------------

async def daily_trading_cycle(connection, account):
    """
    Full daily logic:
      1. Force-close positions held >= MAX_TRADE_DAYS trading days.
      2. Count open slots.
      3. Score all symbols with the LSTM model.
      4. Filter (opposite direction, min prediction).
      5. Rank by |prediction| descending.
      6. Open new trades to fill available slots (up to MAX_POSITIONS).
    """
    log.info("=" * 60)
    log.info("STARTING DAILY TRADING CYCLE")
    log.info("=" * 60)

    # ---- Fetch current account state ----
    terminal_state = connection.terminal_state
    positions      = terminal_state.positions          # list of open positions
    account_info   = terminal_state.account_information

    if account_info is None:
        log.error("Could not retrieve account information. Aborting cycle.")
        return

    free_margin  = account_info.get('freeMargin', account_info.get('equity', 0))
    log.info(f"Account | Free Margin: ${free_margin:,.2f} | Open positions: {len(positions)}")

    # ---- Step 1: Force-close stale positions ----
    log.info(f"Step 1: Checking positions older than {MAX_TRADE_DAYS} trading days...")
    for pos in list(positions):
        open_time  = pos.get('time', '')
        days_open  = count_trading_days_open(open_time)
        if days_open >= MAX_TRADE_DAYS:
            sym = pos.get('symbol', '?')
            log.info(f"  Closing #{pos.get('id')} ({sym}) — {days_open} trading days old.")
            try:
                await connection.close_position(pos['id'])
                log.info(f"  Closed #{pos.get('id')} successfully.")
            except Exception as e:
                log.error(f"  Failed to close #{pos.get('id')}: {e}")

    # Re-fetch positions after potential closures
    await asyncio.sleep(2)
    positions    = connection.terminal_state.positions
    current_count = len(positions)
    slots_available = MAX_POSITIONS - current_count
    already_traded  = {base_symbol(p['symbol']) for p in positions}

    log.info(f"Step 2: Open: {current_count}/{MAX_POSITIONS} | Slots: {slots_available}")

    if slots_available <= 0:
        log.info("No open slots — skipping symbol scoring.")
        log.info("Cycle complete.")
        return

    # ---- Step 3: Score all symbols ----
    log.info(f"Step 3: Scoring {len(SYMBOL_LIST)} symbols with LSTM model...")
    candidates = []   # (broker_symbol, prediction_pct, is_long)

    for raw in SYMBOL_LIST:
        if raw in already_traded:
            log.debug(f"  {raw}: already in position — skip")
            continue

        prediction = predict_symbol(raw)
        if prediction is None:
            log.warning(f"  {raw}: prediction failed — skip")
            continue

        if abs(prediction) < MIN_PREDICTION_PCT:
            log.debug(f"  {raw}: |pred|={abs(prediction):.2f}% below threshold — skip")
            continue

        if OPPOSITE_DIR_REQUIRED:
            yest = get_yesterday_change(raw)
            if yest is None:
                log.warning(f"  {raw}: cannot get yesterday change — skip")
                continue
            if prediction > 0 and yest >= 0:
                log.debug(f"  {raw}: LONG but yesterday was UP — skip")
                continue
            if prediction < 0 and yest <= 0:
                log.debug(f"  {raw}: SHORT but yesterday was DOWN — skip")
                continue

        is_long = prediction > 0
        sym_broker = broker_symbol(raw)
        log.info(
            f"  {raw} -> {sym_broker}: pred={prediction:+.2f}% | "
            f"{'LONG' if is_long else 'SHORT'} — PASSED"
        )
        candidates.append((sym_broker, prediction, is_long))

    # ---- Step 4: Rank by |prediction| descending ----
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    log.info(f"Step 4: {len(candidates)} candidate(s) after all filters.")

    # ---- Step 5: Open trades ----
    opened = 0
    budget_per_trade = free_margin / MAX_POSITIONS

    for sym_broker, prediction, is_long in candidates:
        if opened >= slots_available:
            break

        try:
            # Get symbol specification for lot/price info
            specs = await account.get_symbol_specification(sym_broker)
            tick  = await account.get_symbol_price(sym_broker)

            entry_price   = tick['ask'] if is_long else tick['bid']
            contract_size = specs.get('contractSize', 1)
            min_lot       = specs.get('minVolume', 0.01)
            max_lot       = specs.get('maxVolume', 1000.0)
            lot_step      = specs.get('volumeStep', 0.01)
            digits        = specs.get('digits', 5)

            lot = calculate_lot_size(
                entry_price, contract_size,
                min_lot, max_lot, lot_step,
                budget_per_trade
            )
            if lot <= 0:
                log.warning(f"  {sym_broker}: lot=0, skipping.")
                continue

            T_pct = abs(prediction) / 100.0
            if abs(prediction) < 3.0:
                T_pct *= UNDER_3_PCT_TP_FACTOR

            if is_long:
                tp_price = round(entry_price * (1.0 + T_pct), digits)
                sl_price = round(entry_price * (1.0 - T_pct * RISK_PROFIT_RATIO), digits)
                result = await connection.create_market_buy_order(
                    sym_broker, lot, sl_price, tp_price,
                    {'comment': f"Bot LONG pred={prediction:.2f}%"}
                )
            else:
                tp_price = round(entry_price * (1.0 - T_pct), digits)
                sl_price = round(entry_price * (1.0 + T_pct * RISK_PROFIT_RATIO), digits)
                result = await connection.create_market_sell_order(
                    sym_broker, lot, sl_price, tp_price,
                    {'comment': f"Bot SHORT pred={prediction:.2f}%"}
                )

            log.info(
                f"  Opened {'LONG' if is_long else 'SHORT'} | {sym_broker} | "
                f"Lots: {lot} | Entry: {entry_price:.{digits}f} | "
                f"TP: {tp_price:.{digits}f} | SL: {sl_price:.{digits}f} | "
                f"pred={prediction:+.2f}%"
            )
            opened += 1

        except Exception as e:
            log.error(f"  Failed to open trade for {sym_broker}: {e}")

    log.info(f"Step 5: Opened {opened} new trade(s).")
    log.info("DAILY CYCLE COMPLETE.")
    log.info("=" * 60)

# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def next_run_time() -> datetime:
    """
    Return next datetime (UTC-aware) when the daily cycle should fire:
    TRADE_HOUR_ET:TRADE_MINUTE_ET Eastern Time, on a weekday.
    """
    et_tz  = pytz.timezone("America/New_York")
    now_et = datetime.now(et_tz)
    target = now_et.replace(
        hour=TRADE_HOUR_ET, minute=TRADE_MINUTE_ET, second=0, microsecond=0
    )
    if now_et >= target:
        target += timedelta(days=1)
    while target.weekday() >= 5:    # skip Saturday / Sunday
        target += timedelta(days=1)
    return target


def seconds_until(target: datetime) -> float:
    now   = datetime.now(pytz.utc).astimezone(target.tzinfo)
    delta = (target - now).total_seconds()
    return max(0.0, delta)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_bot():
    """Main async bot loop. Runs one cycle per trading day."""
    log.info("MetaTraderBot (MetaAPI edition) starting up...")

    # Load model
    load_model_and_scaler()

    # Connect to MetaAPI
    log.info("Connecting to MetaAPI...")
    api     = MetaApi(METAAPI_TOKEN)
    account = await api.metatrader_account_api.get_account(METAAPI_ACCOUNT_ID)

    log.info("Deploying/waiting for account to be ready...")
    await account.deploy()
    await account.wait_connected()

    connection = account.get_streaming_connection()
    await connection.connect()
    await connection.wait_synchronized()
    log.info("MetaAPI connection synchronised. Bot is ready.")

    while True:
        try:
            nxt       = next_run_time()
            wait_secs = seconds_until(nxt)
            log.info(
                f"Next cycle at {nxt.strftime('%Y-%m-%d %H:%M:%S %Z')} "
                f"(in {wait_secs/3600:.2f} hours)"
            )

            # Sleep in 60-second chunks
            slept = 0.0
            while slept < wait_secs:
                await asyncio.sleep(min(60, wait_secs - slept))
                slept += 60

            await daily_trading_cycle(connection, account)

        except KeyboardInterrupt:
            log.info("Bot interrupted by user.")
            await connection.close()
            break
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            log.info("Waiting 60 seconds before resuming...")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(run_bot())
