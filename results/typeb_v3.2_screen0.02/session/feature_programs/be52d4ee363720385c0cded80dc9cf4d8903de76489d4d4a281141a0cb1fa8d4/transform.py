def transform(frame):
    cell = frame['fault_mode_cell_1h']
    bank = frame['fault_mode_bank_1h']
    device = frame['fault_mode_device_1h']
    broad = bank + device
    log_cell = np.log1p(cell)
    log_broad = np.log1p(broad)
    fe_cell_broad_log_ratio = log_cell - log_broad
    fe_cell_broad_diff = cell - broad
    fe_cell_broad_sum = cell + broad
    fe_cell_broad_interaction = cell * broad
    return pd.DataFrame({'fe_cell_broad_log_ratio': fe_cell_broad_log_ratio, 'fe_cell_broad_diff': fe_cell_broad_diff, 'fe_cell_broad_sum': fe_cell_broad_sum, 'fe_cell_broad_interaction': fe_cell_broad_interaction}, index=frame.index)