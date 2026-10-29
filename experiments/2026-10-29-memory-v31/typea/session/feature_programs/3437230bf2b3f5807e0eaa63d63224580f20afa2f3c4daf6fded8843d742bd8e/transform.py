def transform(frame):
    denom_cols = frame['raw_life_distinct_cols'].where(frame['raw_life_distinct_cols'] > 0)
    denom_rows = frame['raw_life_distinct_rows'].where(frame['raw_life_distinct_rows'] > 0)
    denom_events = frame['raw_life_events'].where(frame['raw_life_events'] > 0)
    fe_new_col_share = frame['raw_life_new_cols_7d'] / denom_cols
    fe_new_row_share = frame['raw_life_new_rows_7d'] / denom_rows
    fe_new_pattern_col_intensity = frame['raw_life_new_patterns_7d'] / denom_cols
    fe_new_pattern_event_intensity = frame['raw_life_new_patterns_7d'] / denom_events
    return pd.DataFrame({'fe_new_col_share': fe_new_col_share, 'fe_new_row_share': fe_new_row_share, 'fe_new_pattern_col_intensity': fe_new_pattern_col_intensity, 'fe_new_pattern_event_intensity': fe_new_pattern_event_intensity}, index=frame.index)