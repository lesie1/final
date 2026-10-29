def transform(frame):
    recent_patterns = frame['raw_life_new_patterns_7d']
    recent_cols = frame['raw_life_new_cols_7d']
    recent_rows = frame['raw_life_new_rows_7d']
    recent_events = frame['raw_7d_events']
    patterns_per_ce = recent_patterns / (1.0 + recent_events)
    cols_per_ce = recent_cols / (1.0 + recent_events)
    cols_per_row = recent_cols / (1.0 + recent_rows)
    patterns_per_col = recent_patterns / (1.0 + recent_cols)
    fe_log_new_patterns_per_ce_7d = np.log1p(patterns_per_ce)
    fe_log_new_cols_per_ce_7d = np.log1p(cols_per_ce)
    fe_log_new_cols_per_new_row_7d = np.log1p(cols_per_row)
    fe_log_new_patterns_per_new_col_7d = np.log1p(patterns_per_col)
    return pd.DataFrame({'fe_log_new_patterns_per_ce_7d': fe_log_new_patterns_per_ce_7d, 'fe_log_new_cols_per_ce_7d': fe_log_new_cols_per_ce_7d, 'fe_log_new_cols_per_new_row_7d': fe_log_new_cols_per_new_row_7d, 'fe_log_new_patterns_per_new_col_7d': fe_log_new_patterns_per_new_col_7d}, index=frame.index)