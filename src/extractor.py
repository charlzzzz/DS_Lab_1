import yfinance as yf
import pandas as pd
pd.set_option('display.max_rows', None)

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "JNJ",
    "WMT", "PG", "MA", "HD", "BAC", "DIS", "PFE", "PEP", "KO", "AVGO",
    "ADBE", "NFLX", "INTC", "CSCO", "XOM", "VZ", "CRM", "ABT", "T", "ACN",
    "CVX", "COST", "MCD", "TXN", "UNH", "ORCL", "NKE", "QCOM", "LLY", "IBM",
    "GE", "UPS", "MDT", "HON", "NEE", "PM", "AMGN", "GS", "BLK", "MS",
    "LOW", "CAT", "DE", "INTU", "MU", "BKNG", "SPGI", "ADP", "ISRG", "VRTX",
    "ADI",  "BDX", "APD", "ZTS", "MMM", "GILD", "PLD", "CSX",
    "NSC", "TMO", "CCI", "MO", "SO", "DUK", "ETN", "ICE", "FDX", "GM",
    "LRCX", "KLAC", "ILMN", "EXC", "AEP", "EL", "MAR", "ROST", "SBAC", "DHR",
    "MCO", "IDXX", "ITW", "ECL", "PAYX", "STZ", "SHW", "REGN", "WELL", "HCA",
    "AON", "PSA", "CME", "CTSH", "FIS", "CNC", "VLO", "EW", "BIIB", "SBUX",
    "MNST", "MTD", "WBA", "D", "PEG", "XEL", "VRSK", "AMP", "HUM", "AFL",
    "CARR", "RSG", "PH", "SRE", "DHI", "WMB", "CMI", "FTNT", "OKE", "ED",
    "NEM", "PGR", "DD", "ROK", "ALGN", "MSCI", "LYB", "PPG", "IQV", "KMB",
    "BAX", "ORLY", "AVB", "DLR", "EQR", "CTAS", "WEC", "ANSS", "CDNS", "KEYS",
    "TSCO", "LHX", "TRV", "CHD", "AEE", "MLM", "VFC", "OTIS", "LUV", "HBAN",
    "EFX", "HIG", "ZBH", "AJG", "TT", "CINF", "UAL", "JBHT", "MTB",
    "PPL", "APA", "ATO", "EMN", "NRG", "BKR", "ALB", "CF", "NTRS", "AES"
]

# Download data
data = yf.download(
    tickers=tickers,
    start='2020-01-01',
    end='2024-01-01',
    group_by='ticker',
    auto_adjust=False,
    progress=True
)

# Optional: flatten the multi-index columns for easier handling
data.columns = data.columns.swaplevel(0, 1)
data.columns.names = ['Attribute', 'Ticker']
data = data.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()

# Save to CSV
data.to_csv('multi_ticker_data.csv', index=False)
print("✅ Saved data to multi_ticker_data.csv")

# Preview
print(data.head())