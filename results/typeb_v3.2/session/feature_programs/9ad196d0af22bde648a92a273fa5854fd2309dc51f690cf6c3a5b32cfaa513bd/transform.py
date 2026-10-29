def transform(frame):
    adj24 = frame['raw_24h_adjacent_dq_events'] / frame['raw_24h_events']
    mult24 = frame['raw_24h_multi_dq_events'] / frame['raw_24h_events']
    adj7 = frame['raw_7d_adjacent_dq_events'] / frame['raw_7d_events']
    mult7 = frame['raw_7d_multi_dq_events'] / frame['raw_7d_events']
    return pd.DataFrame({'fe_adjacent_dq_share_24h': adj24, 'fe_multi_dq_share_24h': mult24, 'fe_adjacent_dq_share_7d': adj7, 'fe_multi_dq_share_7d': mult7}, index=frame.index)