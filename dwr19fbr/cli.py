"""
Command line interface (cli) for CCO download and processing


"""


import os
import sys
import math
import logging
import datetime
import tempfile

import click
import tqdm

@click.group()
def cli():
    pass

@cli.command('download-cco')
@click.argument('infile', type=click.Path(file_okay=True, dir_okay=False, exists=True))
@click.option('-o', '--out-folder', type=click.Path(file_okay=False, writable=True), required=True)
@click.option('-n', '--nproc', default=None, type=int, help='Number of parallel processes')
def download_cco(infile, out_folder, nproc):
    from .download_module import download_cco_data 
    """Download and unzip Channel Coastal Observatory wave buoy data."""
    os.makedirs(out_folder, exist_ok=True)

    logfile = os.path.join(out_folder, f'download_cco_{datetime.datetime.today():%Y%m%dT%H%M%S}.log')
    setup_file_logger(logfile)

    try:
        download_cco_data(infile, out_folder, nproc=nproc)
    except Exception as e:
        click.echo(f'Error during download: {e}', err=True)
        raise
    else:
        click.echo('Download and unzip completed successfully.')
    finally:
        click.echo(f'Log file saved to: {logfile}')
