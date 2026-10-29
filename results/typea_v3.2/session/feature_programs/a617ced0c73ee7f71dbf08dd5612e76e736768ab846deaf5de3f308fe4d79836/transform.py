def transform(frame):
    adj_share = frame['raw_24h_adjacent_dq_events'] / (frame['raw_24h_events'] + 1)
    multi_share = frame['raw_24h_multi_beat_events'] / (frame['raw_24h_events'] + 1)
    adj_x_spread = adj_share * frame['fe_spread_error_intensity']
    multi_x_pattern = multi_share * frame['fe_recent_new_pattern_rate']
    return pd.DataFrame({'fe_adj_dq_share_24h': adj_share, 'fe_multi_beat_share_24h': multi_share, 'fe_adj_dq_share_x_spread': adj_x_spread, 'fe_multi_beat_share_x_new_pattern': multi_x_pattern}, index=frame.index)