def transform(frame):
    cols_per_cell_24h = frame['raw_24h_distinct_cols'] / (frame['raw_24h_distinct_cells'] + 1)
    cols_per_cell_7d = frame['raw_7d_distinct_cols'] / (frame['raw_7d_distinct_cells'] + 1)
    cols_recent_increase = frame['raw_24h_distinct_cols'] - frame['raw_6h_distinct_cols']
    cols_event_log_ratio = np.log1p(frame['raw_24h_distinct_cols']) - np.log1p(frame['raw_24h_events'])
    return pd.DataFrame({'fe_cols_per_cell_24h': cols_per_cell_24h, 'fe_cols_per_cell_7d': cols_per_cell_7d, 'fe_cols_recent_increase_24h_minus_6h': cols_recent_increase, 'fe_cols_event_log_ratio_24h': cols_event_log_ratio}, index=frame.index)