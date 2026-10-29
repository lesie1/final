def transform(frame):
    denom = frame['all_ce_count_1h'].clip(lower=1)
    b1_share = frame['burst_count_1_1h'] / denom
    b23_share = (frame['burst_count_2_1h'] + frame['burst_count_3_1h']) / denom
    storm_1h_24h = frame['ce_storm_count_1h'].rolling('24h', min_periods=1).sum()
    storm_15m_24h = frame['ce_storm_count_15m'].rolling('24h', min_periods=1).sum()
    return pd.DataFrame({'fe_burst1_share_1h': b1_share, 'fe_burst23_share_1h': b23_share, 'fe_storm_1h_24h': storm_1h_24h, 'fe_storm_15m_24h': storm_15m_24h}, index=frame.index)