def transform(frame):
    denom6 = np.maximum(frame['raw_6h_events'], 1.0)
    share6 = frame['raw_6h_multi_beat_events'] / denom6
    denom24 = np.maximum(frame['raw_24h_events'], 1.0)
    share24 = frame['raw_24h_multi_beat_events'] / denom24
    span6 = frame['raw_6h_max_beat_interval'] * share6
    span24 = frame['raw_24h_max_beat_interval'] * share24
    return pd.DataFrame({'fe_6h_multi_beat_share': share6, 'fe_24h_multi_beat_share': share24, 'fe_6h_beat_span_weighted': span6, 'fe_24h_beat_span_weighted': span24}, index=frame.index)