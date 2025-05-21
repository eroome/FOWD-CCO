"""
cco.py

CCO input file processing into FOWD datasets.

Following download and pre-processing using (pre-process_cco.py),
input files are required to contain the following variables:
- time
- z_displacement
- qc_time
- qc_flag

Optional variables include:
- spectra_energy_density
- spectra_mean_wave_direction
- spectra_directional_spread
- spectra_start_time
- spectra_frequency
- spectra_peak_direction
    
"""

import sys
import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing
import concurrent.futures
import tqdm
import glob
import functools

from .constants import SSH_REFERENCE_INTERVAL
from .output import write_records
from .processing import compute_wave_records, read_pickle_outfile_chunks, read_pickle_statefile


logger = logging.getLogger(__name__)


EXTRA_METADATA = dict(
    contributor_name='Channel Coastal Observatory (CCO)',
    contributor_role='Station operation, station funding',
    acknowledgment=(
        'Channel Coastal Observatory (the National Network of Regional Coastal Monitoring Programmes)'
        'The instrument that collected this dataset was funded by Department for Environment, Food & '
        'Rural Affairs, UK Government.'
    ),
)


# dataset-specific helpers
def mask_invalid(data, mask_duration='30min'):
    """ Mask all displacement data in time window preceding error flags.
    CCO flagging system:
    0    All data pass 
    1    Either HS or TZ fail, so all data fail (except SST) 
    2    TP fail + derivatives 
    3    Dir fail + derivatives 
    4    Spread fail + derivatives 
    5    Not used
    6    Re-processed due to breaking waves 
    7    Buoy off location 
    8    SST fail 
    9    Missing data """
    
    selected_flags = [1, 2, 3, 4, 7]

    # Convert time variables to seconds
    xyz_time_s = data['xyz_time'].values.astype('int64') / 1e9
    qc_time_s  = data['qc_time'].values.astype('int64') / 1e9
    qc_flags   = data['qc_flag'].values.flatten()

    # Count how many of each flag exist
    unique, counts = np.unique(qc_flags, return_counts=True)

    # Extract bad times
    bad_times = qc_time_s[np.isin(qc_flags, selected_flags)]
    
    # Compute mask window in seconds
    mask_duration_s = pd.to_timedelta(mask_duration).total_seconds()

    # Create time-based mask
    mask = np.zeros_like(xyz_time_s, dtype=bool)
    for t in bad_times:
        mask |= (xyz_time_s >= t - mask_duration_s) & (xyz_time_s <= t)

    for comp in ['z_displacement']:
        if comp in data:
            # Apply mask
            data[comp] = data[comp].where(~xr.DataArray(mask, dims='xyz_time'))
    return data


def add_surface_elevation(data):
    """Add surface elevation variable to xarray Dataset."""

    dt = float(1 / data.sampling_rate)
    window_size = int(60 * SSH_REFERENCE_INTERVAL / dt)

    data['mean_displacement'] = (
        data['z_displacement'].rolling(
            {'xyz_time': window_size},
            min_periods=60,
            center=False
        ).mean()
    )

    data['surface_elevation'] = data['z_displacement'] - data['mean_displacement']
    return data


class InvalidFile(Exception):
    pass


def get_input_data(filepath):
    """Read input file as xarray Dataset."""

    required_vars = ['xyz_time', 'z_displacement','qc_time','qc_flag'] 
    optional_vars = ['spectra_energy_density', 'spectra_mean_wave_direction', 'spectra_directional_spread',
                    'spectra_start_time','spectra_frequency','spectra_peak_direction']
    
    def drop_unnecessary(ds):
        # check whether all variables are present
        if any(var not in ds.variables for var in required_vars):
            raise RuntimeError(
                f'Input file does not contain all required variables ({required_vars})'
            )
            
        for v in ds.variables:
            if v not in required_vars + optional_vars:  
                ds = ds.drop(v)
        
        return ds

    data = drop_unnecessary(xr.open_dataset(filepath))
    data = mask_invalid(data)
    data = add_surface_elevation(data)
    return data


def get_wave_records(filepath, out_folder, qc_outfile=None):
    """Process a single file and write results to pickle file."""
    filename = os.path.basename(filepath)
    outfile = os.path.join(out_folder, f'{filename}.waves.pkl')
    statefile = os.path.join(out_folder, f'{filename}.state.pkl')

    # parse file into xarray Dataset
    data = get_input_data(filepath)

    # extract relevant quantities from xarray dataset
    t = np.ascontiguousarray(data['xyz_time'].values)
    z = np.ascontiguousarray(data['z_displacement'].values)
    z_normalized = np.ascontiguousarray(data['surface_elevation'].values)

    meta_args = dict(
        filepath=filepath,
        uuid=data.attrs['id'],
        latitude=data.attrs['latitude'],
        longitude=data.attrs['longitude'],
        water_depth=np.float64(data.attrs['water_depth']),
        sampling_rate=data.attrs['sampling_rate']
    )
    
    try:
        direction_args = dict(
            direction_time=np.ascontiguousarray(data.spectra_start_time.values),
            direction_frequencies=np.ascontiguousarray(data.spectra_frequency.values),
            direction_spread=np.ascontiguousarray(data.spectra_directional_spread.values),
            direction_mean_direction=np.ascontiguousarray(data.spectra_mean_wave_direction.values),
            direction_energy_density=np.ascontiguousarray(data.spectra_energy_density.values),
            direction_peak_direction=np.ascontiguousarray(data.spectra_peak_direction.values),
        )
        
    except (AttributeError, KeyError):
        direction_args = None

    del data  # reduce memory pressure

    compute_wave_records(
        t, z, z_normalized, outfile, statefile, meta_args, direction_args=direction_args,
        qc_outfile=qc_outfile
    )
    
    if direction_args:
        inc_direction = True
    else:
        inc_direction = False

    return outfile, statefile, meta_args['uuid'], inc_direction


def process_single_file(file_path, out_folder):

    station_id = os.path.splitext(os.path.basename(file_path))[0]
    qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')
    
    logger = logging.getLogger(__name__)
    logger.info('Starting processing for file %s', station_id)

    result_file, state_file, station_code, inc_direction = get_wave_records(file_path, out_folder=out_folder, qc_outfile=qc_outfile)

    if result_file is None or state_file is None:
        logger.warning('Processing skipped for file %s', file_path)
        return None

    num_waves = 0
    for record_chunk in read_pickle_outfile_chunks(result_file):
        if record_chunk:
            num_waves += len(record_chunk['wave_id_local'])

    if not num_waves:
        logger.warning('No data found in file %s', file_path)
        return None

    qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

    logger.info('Processing finished for file %s', station_id)
    logger.info('  Found %s waves', num_waves)
    logger.info('  Number of QC flags fired:')
    for key, val in qc_flags_fired.items():
        logger.info(f'      {key} {val:>6d}')

    result_generator = filter(None, read_pickle_outfile_chunks(result_file))
    out_file = os.path.join(out_folder, f'fowd_{station_id}.nc')
    logger.info('Writing output to %s', out_file)
    
    write_records(
        result_generator, out_file, station_code,
        include_direction=inc_direction, extra_metadata=EXTRA_METADATA,
    )

    return station_id


def process_cco(input_folder, out_folder, nproc=None):
    input_folder = os.path.normpath(input_folder)
    assert os.path.isdir(input_folder)

    station_files = sorted(glob.glob(os.path.join(input_folder, '*.nc')))
    num_inputs = len(station_files)

    if num_inputs == 0:
        raise RuntimeError('Given input folder does not contain any valid station files')

    if nproc is None:
        nproc = multiprocessing.cpu_count()
    nproc = min(nproc, num_inputs)

    logger = logging.getLogger(__name__)
    logger.info('Starting processing of %s station files with %s processors', num_inputs, nproc)

    with tqdm.tqdm(total=num_inputs, desc="Processing stations", unit="file") as pbar:
        for f in station_files:
            result = process_single_file(f, out_folder) 
            if result:
                pbar.update(1)


    logger.info("All processing complete.")
    
    
    
def process_cco_parallel(station_folder, out_folder, nproc=None):
    """Process all deployments of a single station.

    Supports processing in parallel (one process per input file).
    """
    station_folder = os.path.normpath(station_folder)
    assert os.path.isdir(station_folder)

    station_id = os.path.basename(station_folder)
    glob_pattern = os.path.join(station_folder, f'{station_id}_*.nc') 
    station_files = sorted(glob.glob(glob_pattern))
    qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')

    num_inputs = len(station_files)
    
    station_code_final = None

    if num_inputs == 0:
        raise RuntimeError('Given input folder does not contain any valid station files')

    if nproc is None:
        nproc = multiprocessing.cpu_count()

    nproc = min(nproc, num_inputs)

    result_files = [None for _ in range(num_inputs)]

    do_work = functools.partial(get_wave_records, out_folder=out_folder, qc_outfile=qc_outfile)
    num_waves_total = 0

    def handle_result(i, result, pbar):
        pbar.update(1)

        result_file, state_file, station_code, inc_direction = result
        filename = station_files[i]

        if result_file is None or state_file is None:
            logger.warning('Processing skipped for file %s', filename)
            return

        nonlocal num_waves_total
        num_waves = 0
        for record_chunk in read_pickle_outfile_chunks(result_file):
            if record_chunk:
                num_waves += len(record_chunk['wave_id_local'])

        num_waves_total += num_waves

        result_files[i] = result_file

        # get QC information
        qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

        # log progress
        num_done = sum(res is not None for res in result_files)
        logger.info(
            'Processing finished for file %s (%s/%s done)', filename, num_done, num_inputs
        )
        logger.info('  Found %s waves', num_waves)
        logger.info('  Number of QC flags fired:')
        for key, val in qc_flags_fired.items():
            logger.info(f'      {key} {val:>6d}')
            
        nonlocal station_code_final
        if station_code_final is None:
            station_code_final = station_code

    pbar_kwargs = dict(
        total=num_inputs, position=nproc, unit='file',
        desc='Processing files', dynamic_ncols=True,
        smoothing=0
    )

    logger.info('Starting processing for station %s (%s input files)', station_id, num_inputs)

    try:
        with tqdm.tqdm(**pbar_kwargs) as pbar:
            if nproc > 1:
                # process deployments in parallel
                with concurrent.futures.ProcessPoolExecutor(nproc) as executor:
                    try:
                        future_to_idx = {
                            executor.submit(do_work, station_file): i
                            for i, station_file in enumerate(station_files)
                        }

                        for future in concurrent.futures.as_completed(future_to_idx):
                            handle_result(future_to_idx[future], future.result(), pbar)

                    except Exception:
                        # abort workers immediately if anything goes wrong
                        for process in executor._processes.values():
                            process.terminate()
                        raise
            else:
                # sequential shortcut
                for i, result in enumerate(map(do_work, station_files)):
                    handle_result(i, result, pbar)

    finally:
        # reset cursor position
        sys.stderr.write('\n' * (nproc + 2))

    logger.info('Processing done')

    if not any(result_files):
        logger.warn('Processed no files - no output to write')
        return

    # write output
    def generate_results():
        current_wave_id = 0
        pbar = tqdm.tqdm(total=num_waves_total, desc='Writing output')

        with pbar:
            for result_file in result_files:
                if result_file is None:
                    continue

                for record_chunk in read_pickle_outfile_chunks(result_file):
                    if not record_chunk:
                        continue

                    # fix local id to be unique for the whole station
                    chunk_size = len(record_chunk['wave_id_local'])
                    record_chunk['wave_id_local'] = np.arange(
                        current_wave_id, current_wave_id + chunk_size
                    )
                    current_wave_id += chunk_size

                    yield record_chunk

                    pbar.update(chunk_size)

    result_generator = generate_results()
    out_file = os.path.join(out_folder, f'fowd_cco_{station_id}.nc')
    logger.info('Writing output to %s', out_file)

    write_records(
        result_generator, out_file, station_code_final,
        include_direction=True, extra_metadata=EXTRA_METADATA,
    )