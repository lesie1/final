def transform(frame):
    denom_7d = frame['raw_7d_events'].clip(lower=1.0)
    denom_life = frame['raw_life_events'].clip(lower=1.0)
    fe_recent_24h_share = frame['raw_24h_events'] / denom_7d
    fe_week_life_share = frame['raw_7d_events'] / denom_life
    fe_log_24h_events = np.log1p(frame['raw_24h_events'])
    fe_24h_vs_7d_daily = frame['raw_24h_events'] - frame['raw_7d_events'] / 7.0
    return pd.DataFrame({'fe_recent_24h_share': fe_recent_24h_share, 'fe_week_life_share': fe_week_life_share, 'fe_log_24h_events': fe_log_24h_events, 'fe_24h_vs_7d_daily': fe_24h_vs_7d_daily}, index=frame.index)