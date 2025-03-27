"""
cco.py

CCO input file processing into FOWD datasets.

Input files are assumed to contain the following variables (after download and pre-processing using pre-process_cco.py):
- time
- displacement

Attributes:
- sampling_rate
- water_depth
- longitude
- latitude

"""

import os
import logging

import numpy as np
import xarray as xr
import multiprocessing
import concurrent.futures
import tqdm
import sys

from .constants import SSH_REFERENCE_INTERVAL
from .output import write_records
from .processing import compute_wave_records, read_pickle_outfile_chunks, read_pickle_statefile

logger = logging.getLogger(__name__)


# dataset-specific helpers

def add_surface_elevation(data):
    """Add surface elevation variable to xarray Dataset."""

    dt = float(1 / data.sampling_rate)
    window_size = int(60 * SSH_REFERENCE_INTERVAL / dt)

    data['mean_displacement'] = (
        data['displacement'].rolling(
            {'time': window_size},
            min_periods=60,
            center=False
        ).mean()
    )

    data['surface_elevation'] = data['displacement'] - data['mean_displacement']
    return data


class InvalidFile(Exception):
    pass


def get_input_data(filepath):
    """Read input file as xarray Dataset."""

    allowed_vars = ['time', 'displacement','waveEnergyDensity', 'waveMeanDirection', 'waveSpread','waveTime','waveFrequency','waveDp'] # added directional variables

    def drop_unnecessary(ds):
        # check whether all variables are present
        if any(var not in ds.variables for var in allowed_vars):
            raise RuntimeError(
                f'Input file does not contain all required variables ({allowed_vars})'
            )

        for v in ds.variables:
            if v not in allowed_vars:
                ds = ds.drop(v)
        
        return ds

    data = drop_unnecessary(xr.open_dataset(filepath))
    data = add_surface_elevation(data)
    return data


#

def get_wave_records(filepath, out_folder, qc_outfile=None):
    """Process a single file and write results to pickle file."""
    filename = os.path.basename(filepath)
    outfile = os.path.join(out_folder, f'{filename}.waves.pkl')
    statefile = os.path.join(out_folder, f'{filename}.state.pkl')

    # parse file into xarray Dataset
    data = get_input_data(filepath)

    # extract relevant quantities from xarray dataset
    t = np.ascontiguousarray(data['time'].values)
    z = np.ascontiguousarray(data['displacement'].values)
    z_normalized = np.ascontiguousarray(data['surface_elevation'].values)

    meta_args = dict(
        filepath=filepath,
        uuid=data.attrs.get('uuid', '<not available>'),
        latitude=data.attrs['latitude'],
        longitude=data.attrs['longitude'],
        water_depth=np.float64(data.attrs['water_depth']),
        sampling_rate=data.attrs['sampling_rate']
    )
    
    direction_args = dict(
        direction_time=np.ascontiguousarray(data.waveTime.values),
        direction_frequencies=np.ascontiguousarray(data.waveFrequency.values),
        direction_spread=np.ascontiguousarray(data.waveSpread.values),
        direction_mean_direction=np.ascontiguousarray(data.waveMeanDirection.values),
        direction_energy_density=np.ascontiguousarray(data.waveEnergyDensity.values),
        direction_peak_direction=np.ascontiguousarray(data.waveDp.values),
    )

    del data  # reduce memory pressure

    compute_wave_records(
        t, z, z_normalized, outfile, statefile, meta_args, direction_args=direction_args,
        qc_outfile=qc_outfile
    )

    return outfile, statefile


def process_file(input_file, out_folder, station_id=None):
    """Process a single generic input file."""

    if station_id is None:
        station_id = os.path.splitext(os.path.basename(input_file))[0]

    qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')

    logger.info('Starting processing for file %s', station_id)

    result_file, state_file = get_wave_records(
        input_file, out_folder=out_folder, qc_outfile=qc_outfile
    )

    if result_file is None or state_file is None:
        logger.warning('Processing skipped for file %s', input_file)
        return

    num_waves = 0
    for record_chunk in read_pickle_outfile_chunks(result_file):
        if record_chunk:
            num_waves += len(record_chunk['wave_id_local'])

    if not num_waves:
        logger.warning('No data found in file %s', input_file)
        return

    # get QC information
    qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

    # log progress
    logger.info(
        'Processing finished for file %s', input_file
    )
    logger.info('  Found %s waves', num_waves)
    logger.info('  Number of QC flags fired:')
    for key, val in qc_flags_fired.items():
        logger.info(f'      {key} {val:>6d}')

    logger.info('Processing done')

    # write output
    result_generator = filter(None, read_pickle_outfile_chunks(result_file))
    out_file = os.path.join(out_folder, f'fowd_{station_id}.nc')
    logger.info('Writing output to %s', out_file)
    write_records(result_generator, out_file, station_id)



def process_folder(input_folder, out_folder, station_id=None):
    """Process all generic input files in a folder. - E.R. edit"""
    assert os.path.isdir(input_folder)

    # List all files in the input folder
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        
        # Skip if it's not a file (e.g., skip subdirectories)
        if not os.path.isfile(file_path):
            continue

        # Derive station_id from each file name
        station_id = os.path.splitext(file_name)[0]
        qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')

        logger.info('Starting processing for file %s', station_id)

        # Process each file
        result_file, state_file = get_wave_records(
            file_path, out_folder=out_folder, qc_outfile=qc_outfile
        )

        if result_file is None or state_file is None:
            logger.warning('Processing skipped for file %s', file_path)
            continue

        num_waves = 0
        for record_chunk in read_pickle_outfile_chunks(result_file):
            if record_chunk:
                num_waves += len(record_chunk['wave_id_local'])

        if not num_waves:
            logger.warning('No data found in file %s', file_path)
            continue

        # Get QC information
        qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

        # Log progress
        logger.info('Processing finished for file %s', file_name)
        logger.info('  Found %s waves', num_waves)
        logger.info('  Number of QC flags fired:')
        for key, val in qc_flags_fired.items():
            logger.info(f'      {key} {val:>6d}')

        # Write output
        result_generator = filter(None, read_pickle_outfile_chunks(result_file))
        out_file = os.path.join(out_folder, f'fowd_{station_id}.nc')
        logger.info('Writing output to %s', out_file)
        write_records(result_generator, out_file, station_id)

    logger.info('Processing complete for all files in folder')




def process_folder_parallel(input_folder, out_folder, nproc=None, station_id=None):
    """Process all generic input files in a folder with parallel processing support."""

    # List all files in the input folder
    station_files = [os.path.join(input_folder, file_name) for file_name in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, file_name))]

    # Check if there are valid files to process
    num_inputs = len(station_files)
    if num_inputs == 0:
        raise RuntimeError('Input folder does not contain any valid files')

    # Determine number of processes
    if nproc is None:
        nproc = multiprocessing.cpu_count()
    nproc = min(nproc, num_inputs)  # Limit to number of input files

    result_files = [None for _ in range(num_inputs)]
    num_waves_total = 0

    def do_work(file_path):
        """Wrapper function to process a single file."""
        station_id = os.path.splitext(os.path.basename(file_path))[0]
        qc_outfile = os.path.join(out_folder, f'fowd_{station_id}.qc.json')
        return get_wave_records(file_path, out_folder=out_folder, qc_outfile=qc_outfile)

    def handle_result(i, result, pbar):
        """Handle results from each file and update progress."""
        pbar.update(1)

        result_file, state_file = result
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

        # Get QC information
        qc_flags_fired = read_pickle_statefile(state_file)['num_flags_fired']

        # Log progress
        num_done = sum(res is not None for res in result_files)
        logger.info('Processing finished for file %s (%s/%s done)', filename, num_done, num_inputs)
        logger.info('  Found %s waves', num_waves)
        logger.info('  Number of QC flags fired:')
        for key, val in qc_flags_fired.items():
            logger.info(f'      {key} {val:>6d}')

    # Progress bar setup
    pbar_kwargs = dict(
        total=num_inputs, unit='file', desc='Processing files',
        dynamic_ncols=True, smoothing=0
    )

    logger.info('Starting processing for %s files in folder %s', num_inputs, input_folder)

    try:
        with tqdm.tqdm(**pbar_kwargs) as pbar:
            if nproc > 1:
                # Parallel processing
                with concurrent.futures.ProcessPoolExecutor(nproc) as executor:
                    try:
                        future_to_idx = {executor.submit(do_work, station_file): i for i, station_file in enumerate(station_files)}
                        for future in concurrent.futures.as_completed(future_to_idx):
                            handle_result(future_to_idx[future], future.result(), pbar)
                    except Exception as e:
                        # Terminate workers immediately on error
                        for process in executor._processes.values():
                            process.terminate()
                        raise e
            else:
                # Sequential processing if nproc is 1
                for i, result in enumerate(map(do_work, station_files)):
                    handle_result(i, result, pbar)

    finally:
        # Reset cursor position after progress bar
        sys.stderr.write('\n' * (nproc + 2))

    logger.info('Processing complete for all files in folder')

    if not any(result_files):
        logger.warning('Processed no files - no output to write')
        return

    # Write output
    def generate_results():
        """Generate results for writing to the output file."""
        current_wave_id = 0
        with tqdm.tqdm(total=num_waves_total, desc='Writing output') as pbar:
            for result_file in result_files:
                if result_file is None:
                    continue
                for record_chunk in read_pickle_outfile_chunks(result_file):
                    if not record_chunk:
                        continue
                    chunk_size = len(record_chunk['wave_id_local'])
                    record_chunk['wave_id_local'] = np.arange(current_wave_id, current_wave_id + chunk_size)
                    current_wave_id += chunk_size
                    yield record_chunk
                    pbar.update(chunk_size)

    result_generator = generate_results()
    out_file = os.path.join(out_folder, 'final_output.nc')
    logger.info('Writing output to %s', out_file)
    write_records(result_generator, out_file)

