def transform(frame):
    top = np.maximum(frame['raw_life_top_cell_share'], 1e-6)
    cells = frame['raw_life_distinct_cells']
    recent_share = frame['raw_life_new_cols_7d'] / (frame['raw_7d_distinct_cols'] + 1)
    focus_recent = frame['raw_life_top_cell_share'] * frame['raw_life_new_cols_7d']
    return pd.DataFrame({'fe_diffuse_per_focus': cells / top, 'fe_wide_row_per_focus': frame['raw_life_max_cols_per_row'] / top, 'fe_recent_col_share': recent_share, 'fe_focus_recent_interaction': focus_recent}, index=frame.index)