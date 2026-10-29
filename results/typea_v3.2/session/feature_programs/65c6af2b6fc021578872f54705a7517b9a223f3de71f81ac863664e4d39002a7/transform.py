def transform(frame):
    life_denom = np.log1p(frame['raw_life_events'].clip(lower=1.0))
    recent_pattern_x_spread = frame['fe_recent_new_pattern_rate'] * frame['fe_spread_error_intensity']
    recent_pattern_per_life = frame['fe_recent_new_pattern_rate'] / life_denom
    spread_per_life = frame['fe_spread_error_intensity'] / life_denom
    recent_colshare_x_spread = frame['fe_recent_new_col_share'] * frame['fe_spread_error_intensity']
    return pd.DataFrame({
        'fe_recent_pattern_x_spread': recent_pattern_x_spread,
        'fe_recent_pattern_per_life': recent_pattern_per_life,
        'fe_spread_per_life': spread_per_life,
        'fe_recent_colshare_x_spread': recent_colshare_x_spread
    }, index=frame.index)