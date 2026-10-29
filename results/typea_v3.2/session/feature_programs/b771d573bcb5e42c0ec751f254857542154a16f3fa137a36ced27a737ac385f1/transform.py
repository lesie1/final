def transform(frame):
    recent_high_burst = frame['burst_count_4_1h'] + frame['burst_count_5_1h'] + frame['burst_count_6_1h'] + frame['burst_count_7_1h'] + frame['burst_count_8_1h']
    high_burst_6h = frame['burst_count_4_6h'] + frame['burst_count_5_6h'] + frame['burst_count_6_6h'] + frame['burst_count_7_6h'] + frame['burst_count_8_6h']
    fe_recent_high_burst_share = recent_high_burst / (1.0 + high_burst_6h)
    fe_recent_storm_share = frame['ce_storm_count_1h'] / (1.0 + frame['ce_storm_count_6h'])
    prior_events_7d_minus_6h = (frame['raw_7d_events'] - frame['raw_6h_events']).clip(0.0)
    recent_ce_rate = frame['raw_6h_events'] / 6.0
    prior_ce_rate = prior_events_7d_minus_6h / 162.0
    fe_recent_ce_accel = recent_ce_rate / (1.0 + prior_ce_rate)
    fe_recent_high_burst_log1p = np.log1p(recent_high_burst)
    return pd.DataFrame({'fe_recent_high_burst_share': fe_recent_high_burst_share, 'fe_recent_storm_share': fe_recent_storm_share, 'fe_recent_ce_accel': fe_recent_ce_accel, 'fe_recent_high_burst_log1p': fe_recent_high_burst_log1p}, index=frame.index)