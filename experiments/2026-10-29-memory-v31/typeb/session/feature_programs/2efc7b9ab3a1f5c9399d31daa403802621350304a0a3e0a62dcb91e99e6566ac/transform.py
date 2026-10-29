def transform(frame):
    gap = past.gap_seconds(frame)
    t = gap.fillna(0).cumsum()
    read_ce_log_positive = np.sign(frame['read_ce_log_num_1h'])
    last_ce_time = (t * read_ce_log_positive).expanding().max()
    time_since_last_read_ce_h = ((t - last_ce_time) / 3600.0).clip(0, 168.0)
    presence_24h = read_ce_log_positive.rolling('24h', min_periods=1).mean()
    presence_6h = read_ce_log_positive.rolling('6h', min_periods=1).mean()
    recent_read_ce_count = frame['read_ce_count_1h'].rolling('6h', min_periods=1).sum()
    return pd.DataFrame({
        'fe_time_since_last_read_ce_h': time_since_last_read_ce_h,
        'fe_read_ce_log_presence_24h': presence_24h,
        'fe_read_ce_log_presence_6h': presence_6h,
        'fe_recent_read_ce_count_6h': recent_read_ce_count
    }, index=frame.index)