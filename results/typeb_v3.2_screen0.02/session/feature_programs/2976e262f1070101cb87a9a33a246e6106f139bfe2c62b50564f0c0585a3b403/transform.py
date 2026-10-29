def transform(frame):
    recent_new_cols = frame['raw_life_new_cols_7d']
    recent_7d_cols = frame['raw_7d_distinct_cols']
    life_events = frame['raw_life_events']
    life_cols = frame['raw_life_distinct_cols']
    fe_log_recent_new_cols_vs_life_events = np.log1p(recent_new_cols) - np.log1p(life_events)
    fe_log_recent_7d_cols_vs_life_events = np.log1p(recent_7d_cols) - np.log1p(life_events)
    fe_log_recent_new_cols_vs_life_cols = np.log1p(recent_new_cols) - np.log1p(life_cols)
    fe_log_recent_7d_cols_vs_life_cols = np.log1p(recent_7d_cols) - np.log1p(life_cols)
    out = pd.DataFrame({
        'fe_log_recent_new_cols_vs_life_events': fe_log_recent_new_cols_vs_life_events,
        'fe_log_recent_7d_cols_vs_life_events': fe_log_recent_7d_cols_vs_life_events,
        'fe_log_recent_new_cols_vs_life_cols': fe_log_recent_new_cols_vs_life_cols,
        'fe_log_recent_7d_cols_vs_life_cols': fe_log_recent_7d_cols_vs_life_cols
    }, index=frame.index)
    return out