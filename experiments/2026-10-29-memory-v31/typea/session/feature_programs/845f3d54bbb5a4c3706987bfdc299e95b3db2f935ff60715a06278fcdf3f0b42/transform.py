def transform(frame):
    total = frame['raw_7d_events']
    new_share = frame['raw_life_new_patterns_7d'] / total.where(total > 0)
    dq_share = frame['raw_7d_multi_dq_events'] / total.where(total > 0)
    product = new_share * dq_share
    log_product = np.log1p(product)
    return pd.DataFrame({'fe_new_pattern_share_7d': new_share, 'fe_multi_dq_share_7d': dq_share, 'fe_newpat_multidq_share_product': product, 'fe_log_newpat_multidq_share_product': log_product}, index=frame.index)