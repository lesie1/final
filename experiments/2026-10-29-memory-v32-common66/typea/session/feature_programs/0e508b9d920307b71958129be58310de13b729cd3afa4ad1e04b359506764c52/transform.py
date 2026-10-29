def transform(frame):
    denom_cols = np.maximum(frame['raw_life_distinct_cols'], 1.0)
    denom_events = np.maximum(frame['raw_life_events'], 1.0)
    recent_col_share = frame['raw_life_new_cols_7d'] / denom_cols
    recent_pattern_rate = frame['raw_life_new_patterns_7d'] / denom_events
    spread = 1.0 - frame['raw_life_top_cell_share']
    error_intensity = np.log1p(frame['error_dq_count_6h'] + frame['error_burst_count_6h'])
    spread_error_interaction = spread * error_intensity
    return pd.DataFrame({
        'fe_recent_new_col_share': recent_col_share,
        'fe_recent_new_pattern_rate': recent_pattern_rate,
        'fe_top_cell_dispersion': spread,
        'fe_spread_error_intensity': spread_error_interaction
    }, index=frame.index)