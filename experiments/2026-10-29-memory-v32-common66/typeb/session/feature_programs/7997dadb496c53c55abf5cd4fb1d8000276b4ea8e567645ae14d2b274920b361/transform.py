def transform(frame):
    spatial = frame['fault_row_num_15m'] + frame['fault_column_num_15m']
    burst = np.maximum(frame['burst_count_3_15m'], frame['burst_count_4_15m'])
    spatial_mean = spatial.rolling('24h', min_periods=1).mean()
    spatial_std = spatial.rolling('24h', min_periods=1).std()
    burst_mean = burst.rolling('24h', min_periods=1).mean()
    burst_std = burst.rolling('24h', min_periods=1).std()
    spatial_z = (spatial - spatial_mean) / spatial_std
    burst_z = (burst - burst_mean) / burst_std
    interaction = spatial_z * burst_z
    spatial_z = spatial_z.where(np.isfinite(spatial_z))
    interaction = interaction.where(np.isfinite(interaction))
    return pd.DataFrame({'fe_spatial_extent_15m': spatial, 'fe_burst34_15m': burst, 'fe_spatial_burst_interaction_15m': interaction, 'fe_spatial_z_15m': spatial_z}, index=frame.index)