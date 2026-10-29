def transform(frame):
    new_col_share = frame['raw_life_new_cols_7d'] / (frame['raw_life_distinct_cols'] + 1)
    new_row_share = frame['raw_life_new_rows_7d'] / (frame['raw_life_distinct_rows'] + 1)
    pattern_per_7d_event = frame['raw_life_new_patterns_7d'] / (frame['raw_7d_events'] + 1)
    pattern_vs_life_event = frame['raw_life_new_patterns_7d'] / (frame['raw_life_events'] + 1)
    return pd.DataFrame({'fe_new_col_share_7d': new_col_share, 'fe_new_row_share_7d': new_row_share, 'fe_pattern_per_7d_event': pattern_per_7d_event, 'fe_pattern_vs_life_event': pattern_vs_life_event}, index=frame.index)