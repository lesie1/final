def transform(frame):
    read_1h = np.maximum(frame['read_ce_count_1h'], 1.0)
    read_15m = frame['read_ce_count_15m']
    short_read = read_15m / read_1h

    burst_1h = np.maximum(frame['error_burst_count_1h'], 1.0)
    burst_15m = frame['error_burst_count_15m']
    short_burst = burst_15m / burst_1h

    raw_24h = np.maximum(frame['raw_24h_events'], 1.0)
    raw_6h = frame['raw_6h_events']
    recent_ratio = raw_6h / raw_24h

    all_1h = np.maximum(frame['all_ce_count_1h'], 1.0)
    storm_1h = frame['ce_storm_count_1h']
    storm_density = storm_1h / all_1h

    return pd.DataFrame({
        'fe_short_term_read_ce_share': short_read,
        'fe_short_term_burst_share': short_burst,
        'fe_6h_to_24h_event_ratio': recent_ratio,
        'fe_1h_storm_density': storm_density
    }, index=frame.index)