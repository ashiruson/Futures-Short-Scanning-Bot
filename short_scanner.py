import os
from datetime import datetime, timezone
import numpy as np
import requests

# Safely load local .env file if it exists (for local testing)
try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  pass

# --- CONFIGURATION ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

GATE_SPOT_API = "https://api.gateio.ws/api/v4/spot/tickers"
GATE_FUTURES_API = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
GATE_CANDLES_API = "https://api.gateio.ws/api/v4/spot/candlesticks"


class GateIoClient:
  """Handles public API interactions with Gate.io for spot and futures."""

  @staticmethod
  def get_futures_contracts():
    try:
      res = requests.get(GATE_FUTURES_API, timeout=10)
      res.raise_for_status()
      return {
          item['name'] for item in res.json() if not item.get('in_delisting', False)
      }
    except Exception as e:
      print(f'Error fetching futures contracts: {e}')
      return set()

  @staticmethod
  def get_market_data():
    try:
      res = requests.get(GATE_SPOT_API, timeout=10)
      res.raise_for_status()
      return res.json()
    except Exception as e:
      print(f'Error fetching market tickers: {e}')
      return []

  @staticmethod
  def get_historical_candles(currency_pair, interval='1h', limit=12):
    """Fetches 12-hour historical OHLCV data."""
    try:
      params = {'currency_pair': currency_pair, 'interval': interval, 'limit': limit}
      res = requests.get(GATE_CANDLES_API, params=params, timeout=10)
      res.raise_for_status()
      candles = res.json()
      if not candles or len(candles) < limit:
        return None

      # Gate candlestick array: [timestamp, volume, close, high, low, open]
      return {
          'lows': [float(c[4]) for c in candles],
          'highs': [float(c[3]) for c in candles],
          'closes': [float(c[2]) for c in candles],
          'volumes': [float(c[1]) for c in candles],
      }
    except Exception:
      return None


class RiskAndPsychologyEngine:
  """Calculates dynamic TP, SL, Trailing Stop Loss, and Trader Psychology targets."""

  @staticmethod
  def calculate_levels(candles, current_price, growth_multiplier, leverage_num):
    lowest_price = min(candles['lows'])
    highest_price = max(candles['highs'])

    # 1. Stop Loss: Set above peak high + volatility buffer based on pump magnitude
    volatility_buffer = 0.02 * (1 + (growth_multiplier - 2.0) * 0.1)
    stop_loss = round(highest_price * (1 + volatility_buffer), 6)

    # Maximum allowed stop loss based on leverage liquidation risk safety rule
    max_allowed_sl = current_price * (1 + (0.8 / leverage_num))
    if stop_loss > max_allowed_sl:
      stop_loss = round(max_allowed_sl, 6)

    # 2. Fib-based Take Profits (Psychological Retracement Levels)
    pump_range = current_price - lowest_price
    tp1 = round(current_price - (pump_range * 0.382), 6)  # 38.2% Retracement
    tp2 = round(current_price - (pump_range * 0.618), 6)  # 61.8% Retracement

    # 3. Dynamic Trailing Stop Parameters
    tsl_activation_price = round(current_price * 0.94, 6)  # Trigger TSL after 6% drop in favor
    tsl_callback_rate = round(min(4.0, max(1.5, growth_multiplier * 0.8)), 1)  # 1.5% to 4.0%

    sl_pct = round(((stop_loss - current_price) / current_price) * 100, 2)
    tp1_pct = round(((current_price - tp1) / current_price) * 100, 2)
    tp2_pct = round(((current_price - tp2) / current_price) * 100, 2)

    return {
        'sl': stop_loss,
        'sl_pct': sl_pct,
        'tp1': tp1,
        'tp1_pct': tp1_pct,
        'tp2': tp2,
        'tp2_pct': tp2_pct,
        'tsl_trigger': tsl_activation_price,
        'tsl_callback': tsl_callback_rate,
    }


class TechnicalAnalyzer:
  """Performs Technical Calculations: RSI, Volume Exhaustion, and Reversal Scoring."""

  @staticmethod
  def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
      return 50.0
    deltas = np.diff(closes)
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
      return 100.0
    rs = up / down
    return float(100 - (100 / (1 + rs)))

  @staticmethod
  def evaluate_short_setup(candles):
    lows = candles['lows']
    closes = candles['closes']
    volumes = candles['volumes']

    lowest_price = min(lows)
    current_price = closes[-1]

    if lowest_price <= 0:
      return None

    growth_multiplier = current_price / lowest_price
    if growth_multiplier < 2.0:
      return None

    rsi = TechnicalAnalyzer.calculate_rsi(closes)
    recent_vol_exhaustion = volumes[-1] < np.mean(volumes[-4:-1]) if len(volumes) >= 4 else False

    reversal_score = 0
    if rsi >= 75:
      reversal_score += 40
    elif rsi >= 68:
      reversal_score += 25

    if recent_vol_exhaustion:
      reversal_score += 30
    if growth_multiplier >= 3.0:
      reversal_score += 30

    return {
        'growth_multiplier': round(growth_multiplier, 2),
        'lowest_price': lowest_price,
        'current_price': current_price,
        'rsi': round(rsi, 1),
        'reversal_score': reversal_score,
        'confirmed': reversal_score >= 50,
    }


class RiskManager:
  """Suggests dynamic safe leverage (X1 to X10) or SKIP."""

  @staticmethod
  def suggest_leverage(reversal_score, growth_multiplier):
    if reversal_score < 50:
      return 'SKIP', 0

    if growth_multiplier > 4.0:
      return 'X2', 2
    elif reversal_score >= 80:
      return 'X7', 7
    elif reversal_score >= 65:
      return 'X5', 5
    else:
      return 'X3', 3


class TelegramNotifier:
  """Sends structured Telegram alerts containing TP/SL & Trailing Stop specs and CTA buttons."""

  @staticmethod
  def send_alert(top_coins):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
      print('Telegram credentials missing. Outputting payload to console:')
      print(top_coins)
      return

    message = '🚨 *HIGH-PROBABILITY SHORT SIGNALS* 🚨\n'
    message += f"⏱ *Timestamp:* `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}`\n\n"

    inline_keyboard = {'inline_keyboard': []}

    for idx, coin in enumerate(top_coins, 1):
      pair_name = coin['pair']
      symbol_clean = pair_name.replace('_USDT', '')
      r = coin['risk_levels']

      message += (
          f"*{idx}. {symbol_clean}* | *Growth: {coin['growth_multiplier']}x*\n"
          f"• *Entry:* `{coin['current_price']}` | *RSI:* `{coin['rsi']}`\n"
          f"• *Leverage:* `{coin['leverage_str']}` (Score:"
          f" `{coin['reversal_score']}/100`)\n"
          f"🛑 *Stop Loss:* `{r['sl']}` (`+{r['sl_pct']}%`)\n"
          f"🎯 *TP1 (38.2%):* `{r['tp1']}` (`-{r['tp1_pct']}%`)\n"
          f"🎯 *TP2 (61.8%):* `{r['tp2']}` (`-{r['tp2_pct']}%`)\n"
          f"🌊 *Trailing Stop:* Activation @ `{r['tsl_trigger']}` | Callback:"
          f" `{r['tsl_callback']}%`\n\n"
      )

      gate_futures_url = f'https://www.gate.io/futures/USDT/{pair_name}'
      inline_keyboard['inline_keyboard'].append(
          [{'text': f'📉 Short {symbol_clean} on Gate.io', 'url': gate_futures_url}]
      )

    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown',
        'reply_markup': inline_keyboard,
    }

    try:
      res = requests.post(
          f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
          json=payload,
          timeout=10,
      )
      res.raise_for_status()
      print('Telegram alert sent successfully!')
    except Exception as e:
      print(f'Failed to send Telegram alert: {e}')


def main():
  print('Starting Gate.io Short Scanner & Risk Engine...')
  futures_contracts = GateIoClient.get_futures_contracts()
  tickers = GateIoClient.get_market_data()

  valid_candidates = []

  for ticker in tickers:
    pair = ticker.get('currency_pair', '')
    if not pair.endswith('_USDT') or pair not in futures_contracts:
      continue

    quote_volume = float(ticker.get('quote_volume', 0) or 0)
    if quote_volume < 500000:
      continue

    candles = GateIoClient.get_historical_candles(pair, interval='1h', limit=12)
    if not candles:
      continue

    setup = TechnicalAnalyzer.evaluate_short_setup(candles)
    if setup and setup['confirmed']:
      leverage_str, leverage_num = RiskManager.suggest_leverage(
          setup['reversal_score'], setup['growth_multiplier']
      )

      if leverage_num == 0:
        continue

      risk_levels = RiskAndPsychologyEngine.calculate_levels(
          candles, setup['current_price'], setup['growth_multiplier'], leverage_num
      )

      valid_candidates.append({
          'pair': pair,
          'growth_multiplier': setup['growth_multiplier'],
          'current_price': setup['current_price'],
          'rsi': setup['rsi'],
          'reversal_score': setup['reversal_score'],
          'leverage_str': leverage_str,
          'risk_levels': risk_levels,
      })

  valid_candidates.sort(key=lambda x: x['growth_multiplier'], reverse=True)
  top_5 = valid_candidates[:5]

  if top_5:
    TelegramNotifier.send_alert(top_5)
  else:
    print('No qualified short signals detected during this scan cycle.')


if __name__ == '__main__':
  main()