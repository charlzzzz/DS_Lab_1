import pandas as pd
import ta
from itertools import islice

# Load data
data = pd.read_csv("multi_ticker_data.csv")

# ✅ Rename columns to lowercase for consistency
data.rename(columns={
    'Close': 'close',
    'High': 'high',
    'Low': 'low',
    'Open': 'open',
    'Adj Close': 'adj_close',
    'Volume': 'volume',
    'Ticker': 'ticker'
}, inplace=True)

features = []

def chunked(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

def add_rolling_and_lagged_features(df):
    lag_features = {
        "rsi": [1, 5],
        "macd": [1],
        "macd_hist": [1, 5],
        "price_ema_dist": [5],
        "adx": [1, 5],
        "cci": [1, 5],
        "stoch_k": [1, 5],
    }

    rolling_features = {
        "rsi": [5],
        "macd_hist": [5, 10],
        "price_ema_dist": [5],
        "obv": [5],
        "atr": [5],
        "adx": [5],
        "cci": [5],
        "stoch_k": [5]
    }

    for feature, lags in lag_features.items():
        if feature in df.columns:
            for lag in lags:
                df[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    for feature, windows in rolling_features.items():
        if feature in df.columns:
            for win in windows:
                df[f"{feature}_roll_mean_{win}"] = df[feature].rolling(win).mean().shift(1)

    return df

symbols = data['ticker'].unique()

for batch_num, batch in enumerate(chunked(symbols, 50), start=1):
    print(f"Processing batch {batch_num} with {len(batch)} symbols...")
    batch_data = data[data['ticker'].isin(batch)].copy()

    for symbol in batch:
        print(f" → Processing symbol: {symbol}")
        df_symbol = batch_data[batch_data['ticker'] == symbol].copy()

        if df_symbol.shape[0] < 30:
            print(f" ⚠ Skipping {symbol}: insufficient data ({df_symbol.shape[0]} rows)")
            continue

        df_symbol['rsi'] = ta.momentum.RSIIndicator(close=df_symbol['close'], window=14).rsi()
        ema = ta.trend.EMAIndicator(close=df_symbol['close'], window=20)
        df_symbol['ema'] = ema.ema_indicator()
        df_symbol['price_ema_dist'] = ((df_symbol['close'] - df_symbol['ema']) / df_symbol['ema']) * 100
        macd = ta.trend.MACD(close=df_symbol['close'])
        df_symbol['macd'] = macd.macd()
        df_symbol['macd_hist'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(close=df_symbol['close'])
        df_symbol['bb_upper'] = bb.bollinger_hband()
        df_symbol['bb_lower'] = bb.bollinger_lband()
        df_symbol['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_symbol['close'], volume=df_symbol['volume']).on_balance_volume()
        df_symbol['atr'] = ta.volatility.AverageTrueRange(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close']).average_true_range()
        df_symbol['return_5d'] = df_symbol['close'].pct_change(periods=5) * 100

        df_symbol['adx'] = ta.trend.ADXIndicator(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close'], window=14).adx()
        df_symbol['cci'] = ta.trend.CCIIndicator(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close'], window=14).cci()
        stoch = ta.momentum.StochasticOscillator(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close'])
        df_symbol['stoch_k'] = stoch.stoch()
        df_symbol['ultimate_osc'] = ta.momentum.UltimateOscillator(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close']).ultimate_oscillator()
        df_symbol['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(high=df_symbol['high'], low=df_symbol['low'], close=df_symbol['close'], volume=df_symbol['volume']).chaikin_money_flow()

        for ind in ['rsi', 'macd', 'macd_hist', 'adx', 'cci', 'stoch_k']:
            if ind in df_symbol.columns:
                df_symbol[f'{ind}_diff'] = df_symbol[ind].diff()

        df_symbol['close_diff']   = df_symbol['close'].diff()
        df_symbol['high_diff']    = df_symbol['high'].diff()
        df_symbol['low_diff']     = df_symbol['low'].diff()
        df_symbol['volume_diff']  = df_symbol['volume'].diff()

        df_symbol['close_pct']    = df_symbol['close'].pct_change() * 100
        df_symbol['high_pct']     = df_symbol['high'].pct_change()  * 100
        df_symbol['low_pct']      = df_symbol['low'].pct_change()   * 100
        df_symbol['volume_pct']   = df_symbol['volume'].pct_change() * 100

        df_symbol['return_20d']  = df_symbol['close'].pct_change(20) * 100
        df_symbol['volatility_20d'] = df_symbol['close'].pct_change().rolling(20).std() * 100

        df_symbol = add_rolling_and_lagged_features(df_symbol)

        threshold = 0.5
        df_symbol['pct_change'] = (df_symbol['close'].shift(-1) - df_symbol['close']) / df_symbol['close'] * 100

        df_symbol['target'] = df_symbol['pct_change'].apply(
            lambda x: 1 if x > threshold else (0 if x < -threshold else -1)
        )

        df_symbol = df_symbol[df_symbol['target'] != -1]
        df_symbol.drop(columns=['pct_change'], inplace=True)

        df_symbol.dropna(inplace=True)

        features.append(df_symbol)

final_df = pd.concat(features, ignore_index=True)
final_df["ticker_id"] = final_df["ticker"].astype("category").cat.codes

output_file = 'features_expanded.csv'
final_df.to_csv(output_file, index=False)
print(f"\n✅ Saved final dataset to '{output_file}' with {final_df.shape[0]} rows and {final_df.shape[1]} columns.")

print(final_df.head())
print(final_df.count())