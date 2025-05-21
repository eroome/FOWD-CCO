"""
preprocessing.py

Aggregate sea states into 100 wave chunks. Add some other variables.
"""

import logging

import numpy as np
import xarray as xr

logger = logging.getLogger('preprocessing')


def apply_mask(ds, dim, mask):
    mask = mask.isel(meta_station_name=0)

    if mask.values.all():
        return ds

    idx = np.where(mask.values)[0]
    return ds.isel(wave_id_local=idx)


def add_direction_difference(ds):
    if 'direction_dominant_direction_in_frequency_interval_2' not in ds.variables:
        logger.info('No directional angle to add')
        return ds

    angle_diff = np.abs(
        ds['direction_dominant_direction_in_frequency_interval_2']
        - ds['direction_dominant_direction_in_frequency_interval_5']
    )
    angle_diff = angle_diff.where(angle_diff < 180, 360 - angle_diff)
    ds['direction_swell_wind_sea_crossing_angle'] = angle_diff
    logger.info('Added swell-wind sea crossing angle variable')
    return ds


def add_water_depth_numbers(ds):
    for ii in ('10m', '30m', 'dynamic'):
        ds[f'sea_state_{ii}_peak_relative_depth_log10'] = np.log10(
            ds['meta_water_depth']
            / ds[f'sea_state_{ii}_peak_wavelength']
        )
        ds[f'sea_state_{ii}_peak_ursell_number_log10'] = (
            np.log10(ds[f'sea_state_{ii}_steepness'])
            - 3 * ds[f'sea_state_{ii}_peak_relative_depth_log10']
        )
    logger.info('Added relative depth and Ursell number variables')
    return ds


def weighted_angular_mean(values, weights):
    norm = sum(weights)
    res_x = sum([w / norm * np.sin(np.radians(q)) for w, q in zip(weights, values)])
    res_y = sum([w / norm * np.cos(np.radians(q)) for w, q in zip(weights, values)])
    return np.degrees(np.arctan2(res_x, res_y)) % 360


def add_dominant_directions(ds):
    if 'direction_dominant_direction_in_frequency_interval_2' not in ds.variables:
        logger.info('No directional angle to add')
        return ds

    ds['direction_dominant_direction'] = weighted_angular_mean(
        [ds[f'direction_dominant_direction_in_frequency_interval_{i}'] for i in (2, 3, 4)],
        [ds[f'sea_state_30m_rel_energy_in_frequency_interval_{i}'] for i in (2, 3, 4)],
    )

    ds['direction_dominant_spread'] = weighted_angular_mean(
        [ds[f'direction_dominant_spread_in_frequency_interval_{i}'] for i in (2, 3, 4)],
        [ds[f'sea_state_30m_rel_energy_in_frequency_interval_{i}'] for i in (2, 3, 4)],
    )
    logger.info('Added dominant direction variables')
    return ds


def scatter_frequencies(ds):
    drop = set()
    drop.add('meta_frequency_band')

    current_vars = ds.variables
    for var in current_vars:
        if var.endswith("_in_frequency_interval"):
            for i in ds["meta_frequency_band"].values:
                ds[f"{var}_{i + 1}"] = ds[var].sel(meta_frequency_band=i)
            drop.add(var)

    ds = ds.drop_vars(drop)
    ds = ds.drop_dims(['meta_frequency_band'])
    logger.info('Scattered frequency intervals')
    return ds


def convert_durations(ds):
    for var in ds.variables:
        if np.issubdtype(ds[var].dtype, np.timedelta64):
            ds[var] = (ds[var] / np.timedelta64(1, 's')).astype('float32')
    logger.info('Converted durations to float')
    return ds


def correct_waveheights(ds):
    correction_factor = 1 / (1 - np.pi ** 2 / 6 / (
        ds['meta_sampling_rate'] * ds['sea_state_30m_mean_period_spectral']
    ) ** 2)
    ds['wave_height'] *= correction_factor
    ds['wave_crest_height'] *= correction_factor
    ds['wave_trough_depth'] *= correction_factor
    for ii in ('10m', '30m', 'dynamic'):
        ds[f'sea_state_{ii}_significant_wave_height_direct'] *= correction_factor
        ds[f'sea_state_{ii}_maximum_wave_height'] *= correction_factor

    logger.info(
        'Corrected wave heights with factor (min, mean, max): '
        f'{correction_factor.min().values:.3f}, '
        f'{correction_factor.mean().values:.3f}, '
        f'{correction_factor.max().values:.3f}'
    )
    return ds


def aggregate_sea_states(ds, agg_waves=100):
    ds_station = ds.isel(meta_station_name=0)
    num_waves = len(ds['wave_id_local'])

    # make sure time is sorted
    assert (np.diff(ds_station['wave_start_time'])
            >= np.timedelta64(0, 's')).all()

    def reshape_record(r):
        return r.isel(
            wave_id_local=slice(None, agg_waves * (num_waves // agg_waves)),
        ).values.reshape(-1, agg_waves)

    time = reshape_record(ds_station['wave_start_time'])
    end_time = reshape_record(ds_station['wave_end_time'])
    rel_waveheight = reshape_record(
        ds_station['wave_height'] /
        ds_station['sea_state_dynamic_significant_wave_height_spectral']
    )
    rel_crestheight = reshape_record(
        ds_station['wave_crest_height'] /
        ds_station['sea_state_dynamic_significant_wave_height_spectral']
    )

    time_cover = (end_time - time).sum(axis=1)
    time_diff = end_time[:, -1] - time[:, 0]

    # require that at most 5% of upcoming waves are missing
    mask = time_cover > 0.95 * time_diff

    max_waveheight = rel_waveheight[mask, :].max(axis=1)
    max_crestheight = rel_crestheight[mask, :].max(axis=1)
    segment_start_times = time[mask, 0]
    segment_end_times = end_time[mask, -1]

    aggregate_idx = np.arange(0, num_waves - agg_waves + 1, agg_waves)[mask]
    ds = ds.isel(wave_id_local=aggregate_idx)

    ds['aggregate_id_local'] = xr.DataArray(
        np.arange(len(aggregate_idx)),
        dims=('wave_id_local',)
    )
    ds = ds.swap_dims({'wave_id_local': 'aggregate_id_local'})

    drop = [v for v in ds.variables if v.startswith('wave_')]
    ds = ds.drop_vars(drop)

    ds[f'aggregate_{agg_waves}_start_time'] = xr.DataArray(
        np.array(segment_start_times).reshape(1, -1),
        dims=('meta_station_name', 'aggregate_id_local')
    )
    ds[f'aggregate_{agg_waves}_end_time'] = xr.DataArray(
        np.array(segment_end_times).reshape(1, -1),
        dims=('meta_station_name', 'aggregate_id_local')
    )
    ds[f'aggregate_{agg_waves}_max_rel_wave_height'] = xr.DataArray(
        np.array(max_waveheight).reshape(1, -1),
        dims=('meta_station_name', 'aggregate_id_local')
    )
    ds[f'aggregate_{agg_waves}_max_rel_crest_height'] = xr.DataArray(
        np.array(max_crestheight).reshape(1, -1),
        dims=('meta_station_name', 'aggregate_id_local')
    )

    logger.info(f'Aggregated {len(aggregate_idx)} sea states')
    return ds


def preprocess(ds, buoy_correction=False, aggregate=False):
    assert isinstance(ds, xr.Dataset)

    if len(ds['meta_station_name']) > 1:
        raise ValueError('cannot preprocess data from multiple stations')

    ds = convert_durations(ds)

    if buoy_correction:
        ds = correct_waveheights(ds)

    ds = scatter_frequencies(ds)
    ds = add_water_depth_numbers(ds)
    ds = add_direction_difference(ds)
    ds = add_dominant_directions(ds)

    if aggregate:
        ds = aggregate_sea_states(ds)

    return ds
