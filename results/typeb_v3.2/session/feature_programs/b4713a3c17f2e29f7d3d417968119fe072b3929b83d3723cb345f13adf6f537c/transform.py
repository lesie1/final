def transform(frame):
    den_15m = frame['all_ce_count_15m'].where(frame['all_ce_count_15m'] > 0)
    fe_bits_per_event_15m = frame['error_bit_count_15m'] / den_15m
    den_1h = frame['all_ce_count_1h'].where(frame['all_ce_count_1h'] > 0)
    fe_bits_per_event_1h = frame['error_bit_count_1h'] / den_1h
    life_den = frame['raw_life_max_err_bits'].where(frame['raw_life_max_err_bits'] > 0)
    fe_recent_severity_ratio = frame['raw_7d_max_err_bits'] / life_den
    fe_hourly_bits_24h_max = frame['error_bit_count_1h'].rolling('24h', min_periods=1).max()
    return pd.DataFrame({'fe_bits_per_event_15m': fe_bits_per_event_15m, 'fe_bits_per_event_1h': fe_bits_per_event_1h, 'fe_recent_severity_ratio': fe_recent_severity_ratio, 'fe_hourly_bits_24h_max': fe_hourly_bits_24h_max}, index=frame.index)