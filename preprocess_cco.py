"""
preprocess-cco.py

Compile displacement [.raw] and spectra [.spt] files (from Datawell Waverider MKIII) into structured netCDF format

Inputs:
    - '1Hz' directory containing '.raw' files (required)
    - 'Wave_Spectra' directory containing '.spt' files (optional)
    - 'CCO_QC' directory containing '.txt' file - half-hourly quality-controlled wave parameters (optional)
    - 'station_metadata.csv' metadata file containing station names, latitude, longitude, etc. (required)

"""

import os
import pandas as pd
import time
import numpy as np
import xarray as xr
import re
import logging 
import netCDF4
from tqdm import tqdm
from datetime import timedelta
from functools import partial
from multiprocessing import Pool
import warnings

logger = logging.getLogger(__name__)

VAR_METADATA = {
    "xyz_time":                     {"units": "nanoseconds since 1980-01-01T00:00:00", "calendar": "proleptic_gregorian", "long_name": "Displacement time", "dims": ("xyz_time",)},
    "z_displacement":               {"units": "m", "long_name": "Vertical displacement", "dims": ("xyz_time",)},
    "y_displacement":               {"units": "m", "long_name": "North displacement", "dims": ("xyz_time",)},
    "x_displacement":               {"units": "m", "long_name": "West displacement", "dims": ("xyz_time",)},
    "spectra_start_time":           {"units": "milliseconds since 1980-01-01T00:00:00", "calendar": "proleptic_gregorian", "long_name": "Spectra data start time", "dims": ("spectra_start_time",)},
    "spectra_energy_density":       {"units": "m^2/Hz", "long_name": "Wave energy density", "dims": ("spectra_start_time", "spectra_frequency")},
    "spectra_mean_wave_direction":  {"units": "degrees", "long_name": "Mean wave direction", "dims": ("spectra_start_time", "spectra_frequency")},
    "spectra_directional_spread":   {"units": "degrees", "long_name": "Directional wave spread at the peak frequency", "dims": ("spectra_start_time", "spectra_frequency")},
    "spectra_skewness":             {"units": "n/a", "long_name": "Skewness of sea surface elevation", "dims": ("spectra_start_time", "spectra_frequency")},
    "spectra_kurtosis":             {"units": "n/a", "long_name": "Kurtosis of sea surface elevation", "dims": ("spectra_start_time", "spectra_frequency")},
    "spectra_peak_direction":       {"units": "s", "long_name": "Wave direction associated with the most energetic waves in the wave spectrum", "dims": ("spectra_start_time",)},
    "spectra_frequency":            {"units": "Hz", "long_name": "Wave frequency bins", "dims": ("spectra_frequency",)},
}


QC_VARS = {
    'qc_latitude':                  ('f4', 'degrees_north', 'Latitude of instrument'),
    'qc_longitude':                 ('f4', 'degrees_east', 'Longitude of instrument'),
    'qc_flag':                      ('i4', None, 'Quality control flag'),
    'qc_significant_wave_height':   ('f4', 'm', 'Mean height of the highest 1/3rd of the waves calculated from wave spectrum (Hm0)'),
    'qc_maximum_wave_height':       ('f4', 'm', 'Largest zero-upcrossing wave'),
    'qc_peak_period':               ('f4', 's', 'Wave period associated with the most energetic waves in the wave spectrum'),
    'qc_zero_upcross_wave_period':  ('f4', 's', 'Spectral zero-upcross period'),
    'qc_peak_wave_direction':       ('f4', 'degrees', 'Wave direction associated with the most energetic waves in the wave spectrum'),
    'qc_directional_spread':        ('f4', 'degrees', 'Directional wave spread at the peak frequency'),
    'qc_sea_surface_temperature':   ('f4', 'degree_Celsius', 'Sea surface temperature'),
}


def delete_folder_contents(parent_folder):
    """Delete unused folders."""
    
    if os.path.exists(parent_folder):
        for item in os.listdir(parent_folder):
            item_path = os.path.join(parent_folder, item)

            if os.path.isdir(item_path):  # Check if it's a folder
                # Delete all files inside the folder
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    for dir in dirs:
                        os.rmdir(os.path.join(root, dir))
                        
        logger.info(f"Deleted unprocessed data: {os.path.basename(parent_folder)}")
    else:
        logger.info("Parent folder does not exist")


def extract_datetime(file_path):
    """Extract datetime from filename (files contain no time information!)"""
    
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}h\d{2}', file_path)
    if match:
        datetime_str = match.group()
        return pd.Timestamp(datetime_str.replace('h', ':'))
    return None


def load_spt(file_path):
    """Load spectral wave data from a .spt file."""
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            spt_data = np.loadtxt(file_path, skiprows=12, delimiter=',')

        # Ensure all columns have exactly 64 rows and 6 columns**
        if spt_data.shape != (64, 6):
            return np.full((64, 6), np.nan), 1   
        else:
            return spt_data, 0
    except (OSError, ValueError):
        return np.full((64, 6), np.nan), 1    

            
def load_raw(file_path):
    """Load displacement data from a .raw file."""
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            raw_data = np.loadtxt(file_path, delimiter=',', dtype=int)
        
        # Ensure all files have the correct number of samples and variables
        if raw_data.shape != (NUM_SAMPLES,4): 
            return np.full((NUM_SAMPLES, 4), np.nan), 1
        else:
            return raw_data, 0
    except (OSError, ValueError):
        return np.full((NUM_SAMPLES, 4), np.nan), 1


def return_spt_ds(spt_data_array, spt_time_array):
    """Generate and return an xarray dataset containing spectral data."""

    num_times, num_frequencies, num_vars = spt_data_array.shape  # Ensure correct shape
    spectral_vars = {}

    # Calculate spectra_peak_direction (peak wave direction at max energy density)
    spectra_energy_density = spt_data_array[:, :, 1]  # Energy density
    spectra_mean_wave_direction = spt_data_array[:, :, 2]  # Mean direction
    
    spectra_peak_direction_indices = np.argmax(spectra_energy_density, axis=1)  # Find index of max energy density for each time step
    spectra_peak_direction = spectra_mean_wave_direction[np.arange(num_times), spectra_peak_direction_indices]  # Extract corresponding mean direction

    # Define variable names
    var_names = ['spectra_energy_density', 'spectra_mean_wave_direction', 'spectra_directional_spread', 'spectra_skewness', 'spectra_kurtosis']

    # Populate spectral_vars dictionary
    for i, var_name in enumerate(var_names):
        spectral_vars[var_name] = (('spectra_start_time', 'spectra_frequency'), spt_data_array[:, :, i+1])

    # Store spectra_peak_direction separately as a 1D array with time dimension
    spectral_vars['spectra_peak_direction'] = ('spectra_start_time', spectra_peak_direction)
    
    # Convert time to nano seconds since time origin
    spt_time_ns_since =  (np.array(spt_time_array) - np.datetime64(TIME_ORIGIN)) // np.timedelta64(1, 'ms')

    # Create xarray Dataset
    spt_ds = xr.Dataset(
        spectral_vars,
        coords={
            'spectra_start_time': spt_time_ns_since,  # Use the provided time list
            'spectra_frequency': SPECTRA_FREQUENCY,  # Frequency indices (0-63)
        })
    spt_ds.coords['spectra_start_time'].attrs['units'] = f"milliseconds since {TIME_ORIGIN}"
    spt_ds.coords['spectra_start_time'].attrs['calendar'] = "proleptic_gregorian"
    return spt_ds 


def process_spt(spt_file_chunk, station_name, prev_spt_time = None):
    """Process a given list of .spt files, ensuring 64 values per timestamp."""
    
    spt_data_list, spt_time_list = [], []   
    missing_files = incomplete_files = 0
    
    for spt_file in sorted(spt_file_chunk):
        spt_time = extract_datetime(spt_file)  # Extract timestamp from filename
        
        # Check time difference between current and previous file (for missing data count)
        if prev_spt_time is None: 
            prev_spt_time = spt_time     
        else:
            time_diff = (spt_time - prev_spt_time).total_seconds() / 60
            if time_diff > 35: # Spectra file timestamps are usually between 30 - 32 minuites apart (this needs some work!)
                num_missing_files = int(time_diff // 35) 
                missing_files += int(num_missing_files)
        
        # Load .spt file data 
        if spt_time is not None:
            spt_data, incomplete_file = load_spt(spt_file)  # Load .spt file data
            incomplete_files += int(incomplete_file) if incomplete_file is not None else 0  # Load .spt file data
            
            if spt_data is not None and spt_data.shape == (64,6):
                # Add data to lists 
                spt_data_list.append(spt_data)
                spt_time_list.extend([np.datetime64(spt_time)]) 
                
        prev_spt_time = spt_time
        
    # Stack into a NumPy arrays
    spt_data_array = np.stack(spt_data_list, axis=0)
    spt_time_array = np.array(spt_time_list).astype('datetime64[ms]')
    
    spt_ds = return_spt_ds(spt_data_array, spt_time_array) 
    
    return spt_ds, prev_spt_time, missing_files, incomplete_files

            
def return_raw_ds(raw_data_array, raw_time_array, station_name):
    """Write the combined month data and time to a xarray dataset."""
    
    df = pd.DataFrame(raw_data_array, columns=['status', 'z_displacement', 'y_displacement', 'x_displacement'])
    zdisp, ydisp, xdisp = df[['z_displacement', 'y_displacement', 'x_displacement']].values.T / 100

    raw_time_ns_since =  (np.array(raw_time_array) - np.datetime64(TIME_ORIGIN)) // np.timedelta64(1, 'ns')

    raw_ds = xr.Dataset(
        {
            'z_displacement': (('xyz_time',), zdisp),
            'y_displacement': (('xyz_time',), ydisp),
            'x_displacement': (('xyz_time',), xdisp),
        },
        coords={
            "xyz_time": raw_time_ns_since,
        })
    raw_ds.coords['xyz_time'].attrs['units'] = f"nanoseconds since {TIME_ORIGIN}"
    raw_ds.coords['xyz_time'].attrs['calendar'] = "proleptic_gregorian"
    return raw_ds

    
def calculate_timesteps(datetime_obj):
    """Calculate timesteps for .raw file"""
    
    # Generate an array of timestamps based on the sample rate
    dt_us = int((1 / SAMPLE_RATE) * 1e6)  # microseconds
    time_deltas_us = np.arange(0, NUM_SAMPLES) * np.timedelta64(dt_us, 'us')
    timestamps = np.datetime64(datetime_obj) + time_deltas_us
    
    return timestamps.astype('datetime64[ns]')


def process_raw(raw_file_chunk, station_name, prev_raw_time = None):    
    """Process raw files for a specific chunk (currently using daily chunks - to avoid exceeding RAM), combining vertical dispalcement data, 
    creating a time variable, and filling missing data with NaNs (to create a continuous timeseries)."""

    raw_data_list, raw_time_list = [], []
    missing_files = incomplete_files = 0
    
    # Handle no data in day chunk (need previous raw time, or else the dataset may start with NaNs) - this is triggered if spt starts before raw
    if not raw_file_chunk and prev_raw_time:
       full_day_times = [prev_raw_time + timedelta(minutes=30 * (i+1)) for i in range(48)]
       raw_time_array = np.array([calculate_timesteps(dt) for dt in full_day_times])
       raw_time_array = raw_time_array.flatten()
       raw_data_array = np.full((48*NUM_SAMPLES, 4), np.nan)
       raw_ds = return_raw_ds(raw_data_array, raw_time_array, station_name)
       prev_raw_time = full_day_times[-1]
       missing_files = 48 
       
       return raw_ds, prev_raw_time, missing_files, incomplete_files
    
    def append_raw(raw_file, raw_data_list, raw_time_list, incomplete_files):
        """
        Load a raw file, append its data and timestamp to the lists,
        update the incomplete_files count, and return the new prev_raw_time
        and incomplete_files.
        """
        # Extract, calculate steps, then append time
        raw_time = extract_datetime(raw_file)
        raw_time_list.append(calculate_timesteps(raw_time))
        
        # Load and append data
        raw_data, inc = load_raw(raw_file)
        raw_data_list.append(raw_data)
        
        incomplete_files += int(inc or 0)

        return raw_time, incomplete_files

    # Main loop (if data is in chunk)
    for raw_file in raw_file_chunk:
        raw_time = extract_datetime(raw_file)

        # Only keep files at :00 or :30 - if not correct timestep, then consider 'incomplete'
        if raw_time.minute not in (0, 30):
            incomplete_files += 1
            continue

        # First valid file
        if prev_raw_time is None:
            prev_raw_time, incomplete_files = append_raw(
            raw_file, raw_data_list, raw_time_list, incomplete_files
            )
            continue

        # Calculate time difference in minutes
        time_diff = (raw_time - prev_raw_time).total_seconds() / 60

        # If exactly 30 minutes, process the file
        if time_diff == 30:
            prev_raw_time, incomplete_files = append_raw(
                raw_file, raw_data_list, raw_time_list, incomplete_files
            )

        # If time gap is >= 60, fill missing files with timesteps and NaN data, then append current file
        elif time_diff >= 60 and time_diff % 30 == 0:
            num_missing_files = int(time_diff // 30) - 1
            raw_data = np.full((num_missing_files, NUM_SAMPLES, 4), np.nan) # filler data
            raw_data_list.extend(raw_data)

            missing_times = [
                calculate_timesteps(prev_raw_time + np.timedelta64(30 * (i + 1), 'm'))
                for i in range(num_missing_files)
            ]
            raw_time_list.extend(missing_times)
            missing_files += num_missing_files
            
            # Append current file
            prev_raw_time, incomplete_files = append_raw(
                raw_file, raw_data_list, raw_time_list, incomplete_files
            )
        else:
            continue

        # Update previous time
        prev_raw_time = raw_time
        
    # Fill any missing files (between last file and end of day chunk)
    if prev_raw_time is not None:
        end_of_day = prev_raw_time.replace(hour=23, minute=30, second=0, microsecond=0)
        time_diff = (end_of_day - prev_raw_time).total_seconds() / 60
    
        if time_diff > 0:
            num_missing_files = int(time_diff // 30)
            raw_data = np.full((num_missing_files, NUM_SAMPLES, 4), np.nan)
            raw_data_list.extend(raw_data)
    
            missing_times = [
                calculate_timesteps(prev_raw_time + timedelta(minutes=30 * (i + 1)))
                for i in range(num_missing_files)
            ]
            raw_time_list.extend(missing_times)
            missing_files += num_missing_files
        prev_raw_time = end_of_day # Make sure we update the end of day time
 
    # Convert lists to arrays
    if raw_data_list:
        raw_data_array = np.vstack(raw_data_list)
        raw_time_array = np.hstack(raw_time_list).T
        raw_ds = return_raw_ds(raw_data_array, raw_time_array, station_name)
    else:
        # Handle empty data case
        raw_ds = None

    return raw_ds, prev_raw_time, missing_files, incomplete_files


def process_raw_spt_files(raw_file_chunk, spt_file_chunk, processed_data_path, station_name, attrs, prev_raw_time, prev_spt_time, year, month, day, last_iteration):
    """Process .raw and .spt files and write to netcdf file"""
    
    raw_missing_files, raw_incomplete_files = 0, 0
    spt_missing_files, spt_incomplete_files = 0, 0

    # Process day chunks in xarray datasets (always process raw day chunks - ensure continuous dataset, fill missing with NaNs)
    raw_ds, prev_raw_time, raw_missing_files, raw_incomplete_files = process_raw(raw_file_chunk, station_name, prev_raw_time)
    
    # Only get spt data if it is available
    if spt_file_chunk: 
        spt_ds, prev_spt_time, spt_missing_files, spt_incomplete_files = process_spt(spt_file_chunk, station_name, prev_spt_time)
    else:
        spt_ds = None
        
    # Only merge data if raw data is available, if no raw data skip chunk 
    if spt_ds or raw_ds:
        ds = xr.merge([d for d in (raw_ds, spt_ds) if d], compat='override')
    else:
        return prev_raw_time, prev_spt_time, raw_incomplete_files, spt_incomplete_files, raw_missing_files, spt_missing_files
    nc_filename = os.path.join(processed_data_path, f"{station_name}.nc")
    
    # Write to netCDF 
    """  Append var data to the netCDF files and add meta data """
    if not os.path.exists(nc_filename):    
        create_nc(nc_filename)    
    with netCDF4.Dataset(nc_filename, 'a') as f:
       
       # Initialize lengths
       current_len_time = f.dimensions['xyz_time'].size
       new_len_time = ds.sizes['xyz_time'] if 'xyz_time' in ds.dims else 0
       current_len_wave = f.dimensions['spectra_start_time'].size if 'spectra_start_time' in f.dimensions else 0
       new_len_wave = ds.sizes['spectra_start_time'] if 'spectra_start_time' in ds.dims else 0

       # Extend xyz_time coordinate
       if 'xyz_time' in ds.coords and new_len_time > 0:
           values = ds.coords['xyz_time'].values
           f.variables['xyz_time'][current_len_time:current_len_time + new_len_time] = values

       # Extend spectra_start_time coordinate
       if 'spectra_start_time' in ds.coords and new_len_wave > 0:
           f.variables['spectra_start_time'][current_len_wave:current_len_wave + new_len_wave] = ds.coords['spectra_start_time'].values

       # Write variables
       for var_name, data_array in ds.data_vars.items():
           if var_name not in f.variables:
               logger.warning(f"[!] Variable '{var_name}' not in NetCDF structure. Skipping.")
               continue

           var = f.variables[var_name]
           dims = data_array.dims

           # Determine slice for writing
           if 'xyz_time' in dims:
               start, end = current_len_time, current_len_time + new_len_time
               slices = [slice(start, end) if dim == 'xyz_time' else slice(None) for dim in var.dimensions]
           elif 'spectra_start_time' in dims:
               start, end = current_len_wave, current_len_wave + new_len_wave
               slices = [slice(start, end) if dim == 'spectra_start_time' else slice(None) for dim in var.dimensions]
           else:
               logger.warning(f"[!] Variable '{var_name}' has no recognized time dimension. Skipping.")
               continue
          
           var[tuple(slices)] = data_array.values

       # Global and variable metadata
       if last_iteration:
           for key, value in attrs.attrs.items():
               f.setncattr(key, value)

           for var_name, metadata in VAR_METADATA.items():
               if var_name in f.variables:
                   for attr, value in metadata.items():
                       f.variables[var_name].setncattr(attr, value)

    return prev_raw_time, prev_spt_time, raw_incomplete_files, spt_incomplete_files, raw_missing_files, spt_missing_files


def create_nc(nc_filename):
    """ Creates strucutred netcdf file """
    with netCDF4.Dataset(nc_filename, 'w') as f:
        # Dimensions from metadata
        f.createDimension('xyz_time', None)
        f.createDimension('spectra_start_time', None)
        f.createDimension('spectra_frequency', len(SPECTRA_FREQUENCY))

        # Coordinate and data variable creation from VAR_METADATA
        for var_name, metadata in VAR_METADATA.items():   
            # Get dims from metadata
            dims = metadata["dims"] 
               
            if var_name in ['xyz_time', 'spectra_start_time']:
                dtype = 'f8'
                var = f.createVariable(var_name, dtype, dims, zlib=True, complevel=3)
                if var_name == 'xyz_time':
                    var.units = metadata.get('units', 'nanoseconds since 1980-01-01T00:00:00')
                if var_name == 'spectra_start_time':
                    var.units = metadata.get('units', 'milliseconds since 1980-01-01T00:00:00')
                var.calendar = metadata.get('calendar', 'proleptic_gregorian')
            elif var_name == 'spectra_frequency':
                # Fill the spectra_frequency with the global values
                var = f.createVariable(var_name, 'f4', dims)
                var[:] = SPECTRA_FREQUENCY  
            else:
                # Set dtype based on units hint or default to f4
                dtype = "f8" if "f8" in str(metadata.get("units", "")) else "f4"
                var = f.createVariable(var_name, dtype, dims, zlib=True)
                
            # Set metadata attributes
            for attr, value in metadata.items():
                var.setncattr(attr, value)


def process_files_by_station(raw_station_path, spt_station_path, qc_dataset_path, processed_data_path, station_name, attrs):
    """ Compile all .raw and .spt files into daily chunks for processing """
    
    start_time = time.time() # Get processing time

    raw_files = sorted([os.path.join(root, file) for root, _, files in os.walk(raw_station_path) for file in files if file.endswith('.raw')])
    
    if len(raw_files) == 0:
        logger.error("No files for the station to process.")
        return

    raw_dates = {f: dt for f in raw_files if (dt := extract_datetime(f))}
    
    # Handle no spt data (not essential)
    if spt_station_path:
        spt_files = sorted([os.path.join(root, file) for root, _, files in os.walk(spt_station_path) for file in files if file.endswith('.spt')])
        spt_dates = {f: dt for f in spt_files if (dt := extract_datetime(f))}
        all_datetimes = list(raw_dates.values()) + list(spt_dates.values())
    else:
        all_datetimes = list(raw_dates.values())
        spt_files = None
    
    # Determine full date range
    start_date = min(all_datetimes).date()
    end_date = max(all_datetimes).date()

    # Create daily chunks regardless of missing data (need to append NaN chunks)
    total_days = (end_date - start_date).days + 1
    day_chunks = [start_date + timedelta(days=i) for i in range(total_days)]

    unique_years = sorted({dt.year for dt in day_chunks})

    logger.info(f"{station_name} archive ({unique_years}) contains: {len(raw_files)} displacement data files & {len(spt_files) if spt_files is not None else '0'} spectra data files")
    logger.info(f"Started sequential processing for {station_name} station...")

    prev_raw_time = prev_spt_time = None
    total_raw_inc = total_raw_mis = total_spt_inc = total_spt_mis = 0

    with tqdm(total=len(day_chunks), desc=f"Processing {station_name}", ncols=100) as pbar:
        for i, current_date in enumerate(day_chunks):
            year, month, day = current_date.year, current_date.month, current_date.day # Get day chunk 
            
            raw_file_chunk = sorted([f for f, dt in raw_dates.items() if dt.date() == current_date])
            if spt_files:
                spt_file_chunk = sorted([f for f, dt in spt_dates.items() if dt.date() == current_date])
            else:
                spt_file_chunk = None

            last_iteration = (i == len(day_chunks) - 1)
            prev_raw_time, prev_spt_time, raw_inc, spt_inc, raw_mis, spt_mis = process_raw_spt_files(
                raw_file_chunk, spt_file_chunk, processed_data_path,
                station_name, attrs, prev_raw_time, prev_spt_time,
                year, month, day, last_iteration
            )

            total_raw_inc += raw_inc
            total_spt_inc += spt_inc
            total_raw_mis += raw_mis
            total_spt_mis += spt_mis

            pbar.update(1)

    # Output file path
    nc_filepath = os.path.join(processed_data_path, f"{station_name}.nc") 
   
    # Get 'all years.txt' qc data file and append data to netcdf (should only be one file with this name)
    if os.path.isdir(qc_dataset_path):
        qc_file = [f for f in os.listdir(qc_dataset_path) if f.endswith('all_years.txt')]

        if len(qc_file) != 0:
            qc_file_path = os.path.join(qc_dataset_path, qc_file[0])
            add_qc_data(qc_file_path, nc_filepath)
        else:
            logger.error(f'Cannot locate {station_name} QC file')
    else:
        logger.error(f'Cannot locate {station_name} QC folder')

    # Print a processing summary
    print_station_summary(station_name, len(raw_files), len(spt_files) if spt_files is not None else 0, len(day_chunks), total_raw_inc, total_spt_inc, total_raw_mis, 
                          total_spt_mis, nc_filepath, start_time)
    
    # Delete data files after processing
    #delete_folder_contents(raw_dataset_path)
    #delete_folder_contents(spt_dataset_path)


def add_qc_data(qc_file_path, nc_path):
    """ Add quality controlled 30 minuite sea state parameters to the nc file (processed by the CCO)"""
    
    df = pd.read_csv(qc_file_path, sep='\t', parse_dates=['Date/Time (GMT)'], dayfirst=False)
    
    # Open the NetCDF file in append mode
    with netCDF4.Dataset(nc_path, 'a') as nc:
        
        time_units = f'seconds since {TIME_ORIGIN}'
        calendar = 'proleptic_gregorian'
        
        # Add qc_time dimension
        if 'qc_time' not in nc.dimensions:
            nc.createDimension('qc_time', None)  # Unlimited
    
        # Add qc_time variable
        if 'qc_time' not in nc.variables:
            qc_time_var = nc.createVariable('qc_time', 'f8', ('qc_time',))
            qc_time_var.units = time_units
            qc_time_var.calendar = calendar
        else:
            qc_time_var = nc.variables['qc_time']
    
        # Function to create variables
        def create_var(name, dtype, units, long_name):
            var = nc.createVariable(name, dtype, ('qc_time',), fill_value=9999)
            if units:
                var.setncattr('units', units)
            var.setncattr('long_name', long_name)

        # Create variables
        for name, (dtype, units, long_name) in QC_VARS.items():
            if name not in nc.variables:
                create_var(name, dtype, units, long_name)
                
        # Ensure datetime format               
        df['Date/Time (GMT)'] = pd.to_datetime(df['Date/Time (GMT)'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
          
        # Convert datetime to numeric
        time_vals = netCDF4.date2num(df['Date/Time (GMT)'].to_list(), units=time_units, calendar=calendar)
    
        # Determine append index
        start_idx = len(nc.variables['qc_time'])
        end_idx = start_idx + len(df)

        # map values from QC txt file to variables in netCDF
        QC_WRITE_MAP = {
            'qc_time': time_vals,
            'qc_latitude': df['Latitude'].values,
            'qc_longitude': df['Longitude'].values,
            'qc_flag': df['Flag'].values,
            'qc_significant_wave_height': df['Hs (Hm0)(m)'].values,
            'qc_maximum_wave_height': df['Hmax (m)'].values,
            'qc_peak_period': df['Tp (s)'].values,
            'qc_zero_upcross_wave_period': df['Tz (Tm)(s)'].values,
            'qc_peak_wave_direction': df['Dirp (degrees)'].values,
            'qc_directional_spread': df['Spread (deg)'].values,
            'qc_sea_surface_temperature': df['SST (deg C)'].values,
        }

        # Write values in a loop using map
        for var_name, values in QC_WRITE_MAP.items():
            nc.variables[var_name][start_idx:end_idx] = values
    
    
def print_station_summary(station_name, raw_count, spt_count, day_count, total_raw_inc, total_spt_inc, total_raw_mis, 
                          total_spt_mis, nc_filepath, start_time):
    """ Write a summary of station processing results"""
    
    logger.info("="*60)
    logger.info(f"Summary Report for Station: {station_name}")
    logger.info("="*60)
    logger.info(f"Day chunks processed                              : {day_count}")
    logger.info(f"Displacement files (.raw) sucessfully processed   : {raw_count-total_raw_inc}/{raw_count}")
    logger.info(f"Displacement files incomplete                     : {total_raw_inc}/{raw_count}")
    logger.info(f"Displacement files missing                        : {total_raw_mis}")
    logger.info(f"Spectra files (.spt) sucessfully processed        : {spt_count-total_spt_inc}/{spt_count}")
    logger.info(f"Spectra files incomplete                          : {total_spt_inc}/{spt_count}")
    logger.info(f"Spectra files missing                             : {total_spt_mis}")
    if os.path.exists(nc_filepath):
        file_size = os.path.getsize(nc_filepath)
        logger.info(f"Output NetCDF size                                : {round(file_size/1000000000,2)} GB")
    logger.info(f"Total processing time                             : {round((time.time() - start_time)/60, 2)} minutes")
    logger.info("="*60 + "\n")


def setup_file_logger(logfile):
    """Set up basic file logger (copied from .logs FOWD).""" 

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
        filemode='w',
        force=True
    )

    # Silence third-party loggers
    logging.getLogger('filelock').setLevel('CRITICAL')
    logging.getLogger('multiprocessing').setLevel(logging.WARNING)


# Function to process each station
def process_station(i, station_metadata, SAMPLE_RATE, logger):
       
    # Extract station name
    station_name = station_metadata["station_name"][i]
    station_code = station_metadata["station_code"][i]
    
    # Create station directory
    unprocessed_data_path = os.path.join(UNPROCESSED_DATA_DIR, station_metadata["station_name"][i])

    # Package attributes 
    attrs = xr.Dataset(attrs = {
        "id":                   f'{station_code}',
        "title":                f'Channel Coastal Observatory wave buoy dataset, station {station_name}',
        "summary":              ('Unprocessed displacement and spectra data, derived from in-situ '
                                 'measurements from the Channel Coastal Observatory wave buoy network.'),
        "date_created":         f"{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}",
        "creator_name":         'Edward Roome',
        "creator_email":        'e.roome@bangor.ac.uk',
        "institution":          'Bangor University',
        "source":               'https://coastalmonitoring.org/',
        "station_name":         station_name,
        "longitude":            station_metadata["lon"][i],
        "latitude":             station_metadata["lat"][i],
        "water_depth":          station_metadata["depth"][i],
        "sampling_rate":        SAMPLE_RATE,
        "spring_tidal_range":   station_metadata["spr_tidal_rng"][i],
        "storm_threshold":      station_metadata["storm_thresh"][i],
        "region":               station_metadata["region"][i],
        "deployment_date":      station_metadata["deployment_date"][i],
    })

    # Create folders to save the .nc files and also specify the .nc outputs:
    os.makedirs(UNPROCESSED_DATA_DIR, exist_ok=True)

    # Specify unprocessed data paths
    raw_dataset_path = os.path.join(unprocessed_data_path, '1Hz')
    spt_dataset_path = os.path.join(unprocessed_data_path, 'Wave_Spectra')
    qc_dataset_path = os.path.join(unprocessed_data_path, 'CCO_QC')

    # Check required 1Hz path
    if not os.path.isdir(raw_dataset_path):
        logger.warning(f"Required directory not found: {station_name} {os.path.basename(raw_dataset_path)}")

    # Optional check for Wave_Spectra
    if not os.path.isdir(spt_dataset_path):
        logger.warning(f"Optional directory not found: {station_name} {os.path.basename(spt_dataset_path)}")
        spt_dataset_path = None

    # Call main processing function
    process_files_by_station(raw_dataset_path, spt_dataset_path, qc_dataset_path, OUT_DIR, station_name, attrs)


# Global variables:    
NPROC = 8 # Number of processors available    
TIME_ORIGIN = '1980-01-01T00:00:00'
SAMPLE_RATE = 1.28  # Datawell Waverider output sample rate (Hz) 
RECORD_LENGTH = 30 * 60 # Length of samples (in seconds) - all output files are 30 minutes long
NUM_SAMPLES = int(SAMPLE_RATE * RECORD_LENGTH)
SPECTRA_FREQUENCY = np.concatenate((np.arange(0.025, 0.1, 0.005), np.arange(0.11,0.59,0.01))) # Define frequency bins

# # Set up multiprocessing logging
# logfile = os.path.join(os.getcwd(), f"pre-process_{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}.log")
# setup_file_logger(logfile)
# logger = logging.getLogger(__name__)
    
# Main program: 
def preprocess_cco(UNPROCESSED_DATA_DIR, OUT_DIR, NPROC):

    global UNPROCESSED_DATA_DIR, OUT_DIR, NPROC
    
    # Load station data (folders that match metadata station name)
    station_metadata_path = os.path.join(UNPROCESSED_DATA_DIR, "station_metadata.csv")
    station_metadata = pd.read_csv(station_metadata_path)
    valid_indices = []

    # Find valid data folders
    for i, station in enumerate(station_metadata["station_name"]):
            station_folder = os.path.join(UNPROCESSED_DATA_DIR, station)
            if os.path.isdir(station_folder):
                valid_indices.append(i)
    logger.info(f"Found {len(valid_indices)} valid station data folders")
    
    # Filter metadata for valid stations
    if not valid_indices:
        logger.error("No valid station data folders found. Exiting.")
        exit()

    filtered_metadata = station_metadata.iloc[valid_indices].reset_index(drop=True)

    # Define the function that will be used for parallel processing
    process_station_partial = partial(process_station, station_metadata=filtered_metadata, SAMPLE_RATE=SAMPLE_RATE, logger=logger)

    # Set up multiprocessing
    if len(valid_indices) < NPROC: # If less data files than processors, then adjust the number of processors
        NPROC = len(valid_indices)
    with Pool(processes=NPROC) as pool:  # Adjust the number of processes as needed
        logger.info(f"Started parallel processing {len(valid_indices)} stations using {NPROC} processors")
        pool.map(process_station_partial, range(len(filtered_metadata)))
