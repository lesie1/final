def transform(frame):
    new_cols = frame['raw_life_new_cols_7d']
    new_rows = frame['raw_life_new_rows_7d']
    new_patterns = frame['raw_life_new_patterns_7d']
    life_events = frame['raw_life_events']
    life_cols = frame['raw_life_distinct_cols']
    recent_cols = frame['raw_7d_distinct_cols']
    fe_recent_cols_per_life_event = np.log1p(new_cols) - np.log1p(life_events)
    fe_recent_rows_per_life_event = np.log1p(new_rows) - np.log1p(life_events)
    fe_recent_patterns_per_life_event = np.log1p(new_patterns) - np.log1p(life_events)
    fe_recent_7d_cols_per_lifetime_cols = np.log1p(recent_cols) - np.log1p(life_cols)
    return pd.DataFrame({'fe_recent_cols_per_life_event': fe_recent_cols_per_life_event, 'fe_recent_rows_per_life_event': fe_recent_rows_per_life_event, 'fe_recent_patterns_per_life_event': fe_recent_patterns_per_life_event, 'fe_recent_7d_cols_per_lifetime_cols': fe_recent_7d_cols_per_lifetime_cols}, index=frame.index)