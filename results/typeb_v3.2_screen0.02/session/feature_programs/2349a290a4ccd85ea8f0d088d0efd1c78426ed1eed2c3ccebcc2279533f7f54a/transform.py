def transform(frame):
    read_ce_7d = frame['read_ce_count_1h'].rolling('7D', min_periods=1).mean()
    log_life_cells = np.log1p(frame['raw_life_distinct_cells'])
    recent_col_ratio = (frame['raw_24h_distinct_cols'] + 1) / (frame['raw_life_distinct_cols'] + 1)
    log_7d_cols = np.log1p(frame['raw_7d_distinct_cols'])
    return pd.DataFrame({'fe_read_ce_7d_mean': read_ce_7d,
                         'fe_log_life_cells': log_life_cells,
                         'fe_recent_col_ratio': recent_col_ratio,
                         'fe_log_7d_cols': log_7d_cols}, index=frame.index)