def transform(frame):
    denom_burst1 = frame['burst_count_1_1h'].where(frame['burst_count_1_1h'] > 0)
    fe_burst1_recent_to_hour = frame['burst_count_1_15m'] / denom_burst1
    prior_burst1 = frame['burst_count_1_1h'] - frame['burst_count_1_15m']
    fe_burst1_recent_minus_prior = frame['burst_count_1_15m'] - prior_burst1
    denom_error = frame['error_burst_count_1h'].where(frame['error_burst_count_1h'] > 0)
    fe_error_burst_recent_to_hour = frame['error_burst_count_15m'] / denom_error
    prior_error = frame['error_burst_count_1h'] - frame['error_burst_count_15m']
    fe_error_burst_recent_minus_prior = frame['error_burst_count_15m'] - prior_error
    return pd.DataFrame({'fe_burst1_recent_to_hour': fe_burst1_recent_to_hour, 'fe_burst1_recent_minus_prior': fe_burst1_recent_minus_prior, 'fe_error_burst_recent_to_hour': fe_error_burst_recent_to_hour, 'fe_error_burst_recent_minus_prior': fe_error_burst_recent_minus_prior}, index=frame.index)