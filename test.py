# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:25:04 2025

@author: edroo
"""


import os
import time
import xarray as xr
import numpy as np
import dask
import platform
import psutil
import importlib

def print_system_info():
    print("==== System Info ====")
    print(f"Platform: {platform.platform()}")
    print(f"CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"RAM (GB): {psutil.virtual_memory().total / 1e9:.2f}")
    print()

def print_package_versions():
    print("==== Package Versions ====")
    for pkg in ['xarray', 'dask', 'netCDF4', 'h5py', 'h5netcdf', 'numpy']:
        try:
            mod = importlib.import_module(pkg)
            print(f"{pkg:10s}: {mod.__version__}")
        except ImportError:
            print(f"{pkg:10s}: NOT INSTALLED")
    print()

def benchmark_open_and_slice(nc_file, slice_size=1000):
    print(f"==== Opening Dataset: {nc_file} ====")
    t0 = time.time()
    ds = xr.open_dataset(nc_file, cache=False, chunks={"wave_id_local": 10_000})
    t1 = time.time()
    print(f"Time to open dataset: {t1 - t0:.2f} seconds")
    print("Encoding:", ds.encoding)
    print("Backend engine used:", ds.attrs.get('Conventions', 'unknown'))

    print("\n==== Basic Slice & Load Benchmark ====")
    wave_len = len(ds['wave_id_local'])
    print(f"Total wave records: {wave_len}")
    slice_end = min(slice_size, wave_len)
    
    t2 = time.time()
    dsi = ds.isel(meta_station_name=0, wave_id_local=slice(0, slice_end)).load()
    t3 = time.time()
    print(f"Time to load first {slice_end} records: {t3 - t2:.2f} seconds")

    print("\n==== Dask Config ====")
    print(dask.config.config)
    print()

    ds.close()

if __name__ == "__main__":
    # TODO: Update with your large .nc file path
    netcdf_path = r"C:\Users\edroo\Documents\Files\Research\PhD Rouge Waves\Data\CCO\fowd_cco_data\fowd_cco_BlakeneyOverfalls.nc"

    if not os.path.exists(netcdf_path):
        print(f"NetCDF file not found: {netcdf_path}")
    else:
        print_system_info()
        print_package_versions()
        benchmark_open_and_slice(netcdf_path)
