"""
cli.py

Entry point for CLI.
"""

import os
import sys
import math
import logging
import datetime
import tempfile

import click
import tqdm

from . import __version__


@click.group('fowd', invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx):
    """The command line interface for the Free Ocean Wave Dataset (FOWD) processing toolkit."""

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return


@cli.command('process-cdip')
@click.argument('CDIP_FOLDER', type=click.Path(file_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '-n', '--nproc', default=None, type=int,
    help='Maximum number of parallel processes [default: number of CPU cores]'
)
def process_cdip(cdip_folder, out_folder, nproc):
    """Process all deployments of a CDIP station into one FOWD output file."""
    from .cdip import process_cdip_station
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_cdip_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_cdip_station(cdip_folder, out_folder, nproc=nproc)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')


@cli.command('process-generic')
@click.argument('INFILE', type=click.Path(dir_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '--station-id', default=None,
    help='Station ID to use in outputs [default: use input file name]'
)
def process_generic(infile, station_id, out_folder):
    """Process a generic netCDF input file into a FOWD output file."""
    from .generic_source import process_file
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_generic_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_file(infile, out_folder, station_id=station_id)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')


   
"Added by E.R. to process single cco file"
@cli.command('process-cco')
@click.argument('INFILE', type=click.Path(file_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '-n', '--nproc', default=None, type=int,
    help='Maximum number of parallel processes [default: number of CPU cores]'
)
def process_cco(infile, out_folder, nproc):
    """Process a generic netCDF input file into a FOWD output file."""
    from .cco import process_cco
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_cco_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_cco(infile, out_folder, nproc)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')  
     
        
     
"Added by E.R. to process multiple cco files in parallel"
@cli.command('process-cco-para')
@click.argument('INFILE', type=click.Path(file_okay=False, readable=True, exists=True))
@click.option(
    '-o', '--out-folder',
    type=click.Path(file_okay=False, writable=True),
    required=True,
)
@click.option(
    '-n', '--nproc', default=None, type=int,
    help='Maximum number of parallel processes [default: number of CPU cores]'
)
def process_cco_para(infile, out_folder, nproc):
    """Process a generic netCDF input file into a FOWD output file."""
    from .cco import process_cco_parallel
    from .logs import setup_file_logger

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_cco_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)

    try:
        process_cco_parallel(infile, out_folder, nproc)
    except Exception:
        click.echo('Error during processing', err=True)
        raise
    else:
        click.echo('Processing finished successfully')
    finally:
        click.echo(f'Log file written to {logfile}')  
        
        
        

@cli.command('run-tests')
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True))
def run_tests(out_folder):
    """Run unit tests and sanity checks."""
    import pytest
    from .sanity.run_sanity_checks import run_all

    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='fowd_sanity_')

    os.makedirs(out_folder, exist_ok=True)

    click.echo('Running unit tests ...')
    exit_code = pytest.main([
        '-x',
        os.path.join(os.path.dirname(__file__), 'tests')
    ])

    click.echo('')
    click.echo('Running sanity checks ...')
    run_all(out_folder)
    click.echo(f'Sanity check results written to {out_folder}')
    click.echo(click.style('Make sure to check whether outputs are as expected.', bold=True))

    if exit_code > 0:
        sys.exit(exit_code)


@cli.command('plot-qc')
@click.argument('QC_INFILE', type=click.Path(dir_okay=False, readable=True))
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True))
def plot_qc(qc_infile, out_folder):
    """Generate plots from QC log files."""
    from .postprocessing import plot_qc

    if out_folder is None:
        out_folder = tempfile.mkdtemp(prefix='fowd_qc_')

    click.echo('Plotting QC records ...')
    plot_qc(qc_infile, out_folder)
    click.echo(f'Results written to {out_folder}')


@cli.command('postprocess')
@click.argument('INPUT_FILES', type=click.Path(dir_okay=True, readable=True), nargs=-1)
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True), required=True)
def postprocess(input_files, out_folder):
    """Filter unreliable measurements from the FOWD output:
        - Low swh 
        - High frequency
        - Low frequency drift  """
        
    import xarray as xr
    
    from .postprocessing import run_postprocessing, CDIP_DEPLOYMENT_BLACKLIST
    from .logs import setup_file_logger
    from .output import write_records
    from .cdip import EXTRA_METADATA as CDIP_EXTRA_METADATA
    from .cco import EXTRA_METADATA as CCO_EXTRA_METADATA

    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(
        out_folder,
        f'fowd_postprocessing_{datetime.datetime.today():%Y%m%dT%H%M%S}.log'
    )
    setup_file_logger(logfile)
    logger = logging.getLogger(__name__)
    
    # To handle directory or file path
    if os.path.isdir(str(input_files[0])): 
        multi_input_files = []
        for dirpath, _, filenames in os.walk(str(input_files[0])):
            for filename in filenames:
                if filename.startswith('fowd') and filename.endswith('.nc'):
                    multi_input_files.append(os.path.join(dirpath, filename))
        pbar = tqdm.tqdm(multi_input_files, desc='Post-processing FOWD files')
    else:   
        pbar = tqdm.tqdm(input_files, desc='Post-processing FOWD files')

    for infile in pbar:
        pbar.set_postfix(file=os.path.basename(infile))

        filename, ext = os.path.splitext(os.path.basename(infile))
        outfile = os.path.join(out_folder, f'{filename}_filtered{ext}')
        print('here - just before ds open')
        with xr.open_dataset(infile, cache=False) as ds:
            logger.info(f'Processing {infile}')
            print('opened dataset')
            station_name = str(ds.meta_station_name.values[0])
            num_records = len(ds['wave_id_local'])

            is_cdip = station_name.startswith('CDIP_')
            if is_cdip and CDIP_DEPLOYMENT_BLACKLIST.get(station_name[5:]) == '*':
                logger.info('All deployments blacklisted, skipping')
                continue
            
            include_direction = 'direction_sampling_time' in ds.variables

            out_metadata = {}
            if is_cdip:
                out_metadata.update(CDIP_EXTRA_METADATA)
            
            is_cco = infile.startswith('fowd_cco')
            if is_cco:
                out_metadata.update(CCO_EXTRA_METADATA)
            
            out_metadata['postprocessing'] = 'filtered'
            out_metadata['postprocessing_input_uuid'] = ds.attrs['uuid']

            num_filtered = {}
            # xarray comes to a crawl for smaller chunks for some reason
            chunk_size = 1_000_000

            record_generator = tqdm.tqdm(
                run_postprocessing(ds, num_filtered, chunk_size=chunk_size),
                total=math.ceil(num_records / chunk_size),
                leave=False
            )

            write_records(
                record_generator,
                outfile, station_name,
                extra_metadata=out_metadata,
                include_direction=include_direction,
            )

            for filter_name, filter_num in num_filtered.items():
                logger.info(f'[{filter_name}]: Filtered {filter_num} seas')

    click.echo(f'Results written to {out_folder}')



@click.command('aggregate_sea_states')
@click.argument('INFILES', type=click.Path(dir_okay=False, exists=True), nargs=-1)
@click.option('-o', '--outdir', type=click.Path(file_okay=False, writable=True), required=True)
@click.option('--aggregate', type=click.Choice(['only', 'both', 'false']), default='both')

def aggregate_sea_states(infiles, outdir, aggregate):
    
    import os
    import logging
    from datetime import datetime

    import tqdm
    import numpy as np
    import xarray as xr

    from aggregate_sea_states import preprocess, aggregate_sea_states
    
    os.makedirs(outdir, exist_ok=True)

    pbar = tqdm.tqdm(infiles)

    logfile = os.path.join(outdir, f'rogue-preprocessing-{datetime.today():%Y%m%d-%H%M%S}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(pbar),
        ]
    )
    logger = logging.getLogger('preprocessing')
    
    def convert_datetimes(df):
        return df.astype(
            {c: 'datetime64[ms]' for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)},
            copy=False
        )

    for f in pbar:
        station_id = os.path.splitext(os.path.basename(f))[0]
        pbar.set_postfix(dict(station=station_id))

        outfile_raw = os.path.join(outdir, f'{station_id}-preprocessed.parquet')
        outfile_agg = os.path.join(outdir, f'{station_id}-preprocessed-agg.parquet')

        if os.path.isfile(outfile_raw) and os.path.isfile(outfile_agg):
            continue

        with xr.open_dataset(f, drop_variables=['wave_raw_elevation']) as ds:
            logger.info(f'Processing {station_id} containing {len(ds.wave_id_local)} seas')
            ds = ds.load()
            ds = preprocess(ds, buoy_correction=True, aggregate=False)

            if len(ds['wave_id_local']) == 0:
                logger.warn('Station contains no valid observations')
                continue

            if aggregate != 'only':
                df = ds.to_dataframe()
                convert_datetimes(df).to_parquet(outfile_raw)

            if aggregate != 'false':
                try:
                    ds = aggregate_sea_states(ds)
                except AssertionError as e:
                    logger.error(f'Failed to aggregate records for {station_id}')
                    logger.error(e)
                    continue
                df = ds.to_dataframe()
                convert_datetimes(df).to_parquet(outfile_agg)

        logger.info('Done')


def entrypoint():
    try:
        cli(obj={})
    except Exception:
        logger = logging.getLogger(__name__)
        logger.exception('Uncaught exception!', exc_info=True)
        raise
