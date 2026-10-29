def transform(frame):
    recent_col_share = frame['raw_life_new_cols_7d'] / np.maximum(frame['raw_life_distinct_cols'], 1)
    recent_row_share = frame['raw_life_new_rows_7d'] / np.maximum(frame['raw_life_distinct_rows'], 1)
    recent_col_share_log = np.log1p(recent_col_share)
    recent_row_share_log = np.log1p(recent_row_share)
    return pd.DataFrame({'fe_recent_col_share': recent_col_share, 'fe_recent_row_share': recent_row_share, 'fe_recent_col_share_log': recent_col_share_log, 'fe_recent_row_share_log': recent_row_share_log}, index=frame.index)