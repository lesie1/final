def transform(frame):
    ce15 = frame['all_ce_count_15m']
    ce1h = frame['all_ce_count_1h']
    base15 = ce15.rolling('6h', min_periods=1).mean().shift(1)
    base1h = ce1h.rolling('6h', min_periods=1).mean().shift(1)
    fe_ce_15m_diff_6h = ce15 - base15
    fe_ce_1h_diff_6h = ce1h - base1h
    fe_ce_15m_rel_6h = fe_ce_15m_diff_6h / np.maximum(base15, 1.0)
    fe_ce_1h_rel_6h = fe_ce_1h_diff_6h / np.maximum(base1h, 1.0)
    return pd.DataFrame({'fe_ce_15m_diff_6h': fe_ce_15m_diff_6h, 'fe_ce_1h_diff_6h': fe_ce_1h_diff_6h, 'fe_ce_15m_rel_6h': fe_ce_15m_rel_6h, 'fe_ce_1h_rel_6h': fe_ce_1h_rel_6h}, index=frame.index)