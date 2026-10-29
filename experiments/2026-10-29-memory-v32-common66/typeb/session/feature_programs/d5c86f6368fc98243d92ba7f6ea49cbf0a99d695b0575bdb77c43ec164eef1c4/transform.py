def transform(frame):
    read_present_15m = frame['read_ce_count_15m'].notna().astype('float')
    read_present_1h = frame['read_ce_count_1h'].notna().astype('float')
    read_share_15m = frame['read_ce_count_15m'] / (frame['all_ce_count_15m'] + 1)
    read_share_1h = frame['read_ce_count_1h'] / (frame['all_ce_count_1h'] + 1)
    return pd.DataFrame({
        'fe_read_ce_present_15m': read_present_15m,
        'fe_read_ce_present_1h': read_present_1h,
        'fe_read_ce_share_15m': read_share_15m,
        'fe_read_ce_share_1h': read_share_1h
    }, index=frame.index)