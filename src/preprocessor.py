def drop_high_collinear_features(final_df):
    features_to_drop = [
        'open', 'high', 'low', 'adj_close', 'volume',
        'ema', 'atr',
        'obv_roll_mean_5', "close_diff",
        'atr_roll_mean_5', 'cci_roll_mean_5',
        'bb_upper', 'bb_lower',
        'adx_lag1', 'adx_lag5', 'adx_roll_mean_5',
        'cci_lag1', 'cci_lag5',
        'rsi_lag1', 'rsi_lag5', 'rsi_roll_mean_5',
        'macd_lag1', 'macd_hist_lag1', 'macd_hist_lag5',
        'macd_hist_roll_mean_5', "price_ema_dist_roll_mean_5", "price_ema_dist",
        'stoch_k', 'stoch_k_lag1', 'stoch_k_lag5',
    ]

    df = final_df.drop(columns=features_to_drop, errors='ignore').copy()

    print("\n✅ Overall target class distribution:")
    print(df['target'].value_counts(normalize=True).rename('proportion'))

    return df