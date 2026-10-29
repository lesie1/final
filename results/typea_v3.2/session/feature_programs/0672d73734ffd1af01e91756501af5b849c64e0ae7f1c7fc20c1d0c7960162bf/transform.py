def transform(frame):
    life_cols = np.maximum(frame['raw_life_distinct_cols'], 1)
    life_cells = np.maximum(frame['raw_life_distinct_cells'], 1)
    recent_cols = frame['raw_7d_distinct_cols']
    recent_cells = frame['raw_7d_distinct_cells']
    recent_rows = np.maximum(frame['raw_7d_distinct_rows'], 1)
    recent_col_share = recent_cols / life_cols
    recent_cell_share = recent_cells / life_cells
    recent_col_density = recent_cols / recent_rows
    active_err = frame['error_dq_count_6h'] + frame['error_burst_count_6h']
    spreading_activity = recent_col_share * np.log1p(active_err)
    return pd.DataFrame({
        'fe_recent_col_share': recent_col_share,
        'fe_recent_cell_share': recent_cell_share,
        'fe_recent_col_density': recent_col_density,
        'fe_spreading_activity': spreading_activity
    }, index=frame.index)