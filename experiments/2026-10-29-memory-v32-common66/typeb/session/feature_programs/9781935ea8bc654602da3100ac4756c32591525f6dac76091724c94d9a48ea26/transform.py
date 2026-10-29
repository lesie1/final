def transform(frame):
    den_row = np.maximum(frame['fault_row_num_1h'], 1.0)
    fe_fault_row_growth_recent = frame['fault_row_num_15m'] / den_row
    den_col = np.maximum(frame['fault_column_num_1h'], 1.0)
    fe_fault_column_growth_recent = frame['fault_column_num_15m'] / den_col
    den_bit = np.maximum(frame['error_bit_count_1h'], 1.0)
    fe_error_bit_growth_recent = frame['error_bit_count_15m'] / den_bit
    past_45m_bit = np.maximum(frame['error_bit_count_1h'] - frame['error_bit_count_15m'], 0.0)
    fe_error_bit_recent_excess = frame['error_bit_count_15m'] - past_45m_bit
    return pd.DataFrame({'fe_fault_row_growth_recent': fe_fault_row_growth_recent, 'fe_fault_column_growth_recent': fe_fault_column_growth_recent, 'fe_error_bit_growth_recent': fe_error_bit_growth_recent, 'fe_error_bit_recent_excess': fe_error_bit_recent_excess}, index=frame.index)