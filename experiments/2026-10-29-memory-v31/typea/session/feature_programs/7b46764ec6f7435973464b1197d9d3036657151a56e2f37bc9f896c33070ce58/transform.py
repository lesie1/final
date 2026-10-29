def transform(frame):
    cols = frame['raw_life_distinct_cols']
    cells = frame['raw_life_distinct_cells']
    density = cells / np.maximum(cols, 1)
    density_change = density.diff(1).rolling('7D', min_periods=1).mean()
    new_cols = frame['raw_life_new_cols_7d']
    new_col_share = new_cols / np.maximum(cols, 1)
    interaction = np.log1p(new_cols) * np.log1p(density)
    return pd.DataFrame({
        'fe_life_cell_col_density': density,
        'fe_life_density_7d_change': density_change,
        'fe_life_new_col_share_7d': new_col_share,
        'fe_life_recent_density_interaction': interaction
    }, index=frame.index)