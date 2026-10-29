def transform(frame):
    life_events = frame['raw_life_events'] + 1.0
    pattern_rate = frame['fe_recent_new_pattern_rate']
    normalized = pattern_rate / life_events
    risky_6h = frame['raw_6h_risky_events']
    risky_24h = frame['raw_24h_risky_events']
    burst_15m = frame['error_burst_count_15m']
    return pd.DataFrame({
        'fe_recent_pattern_life_norm': normalized,
        'fe_recent_pattern_x_6h_risky': normalized * risky_6h,
        'fe_recent_pattern_x_24h_risky': normalized * risky_24h,
        'fe_recent_pattern_x_15m_burst': normalized * burst_15m
    }, index=frame.index)