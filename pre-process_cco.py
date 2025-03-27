"""
pre-process_cco.py

Download and pre-process .raw and .spt files from https://coastalmonitoring.org/
"""

# 24/03 E.R. To Do:
# - work on logging (somewhat working)
# - revist time variable format (seems to auto format when using xarray - more checks needed)
# - split this script for ease of use
# - import logger function  from .logs FOWD!


import os
import pandas as pd
import patoolib
import time
import numpy as np
import xarray as xr
import re
import multiprocessing
from itertools import repeat
import pickle
import logging 
import calendar 

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC



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
                #os.rmdir(item_path)  # Delete the now-empty folder
                #print(f"Deleted folder: {item_path}")
        print("All folders inside have been deleted.")
    else:
        print("Parent folder does not exist.")


# Function to only proceed once the download is finished:
def wait_for_download(download_folder, timeout=300):
    """Wait for the latest downloaded file and rename it."""

    seconds = 0
    while seconds < timeout:
        files = os.listdir(download_folder)
        zip_files = [f for f in files if f.endswith(".zip")]  # Only check for .zip files

        if zip_files:
            latest_file = max([os.path.join(download_folder, f) for f in zip_files], key=os.path.getctime)

            return True

        time.sleep(1)
        seconds += 1

    print("Download timeout or file not found.")
    return False


def unzip_year(zip_files, year_data_folders):
    """ Unzip the year data folders."""
    patoolib.extract_archive(zip_files, outdir=year_data_folders) # need to use patoolib not zipfile
    print(f"Successfully processed {os.path.basename(zip_files)}.")
    # Delete the archive after extraction
    os.remove(zip_files)
    print(f"deleted {os.path.basename(zip_files)}")


def unzip_nested_archives(input_folder):
    """Extracts all archives in a folder, including the nested archives."""

    # Iterate through all files in the folder
    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)

        # Check if it's a compressed archive
        if file.endswith((".zip", ".tar", ".rar", ".7z", ".gz", ".bz2", ".xz")):
            print(f"Extracting: {file}")

            # Extract the archive in the same folder
            patoolib.extract_archive(file_path, outdir=input_folder)

            # Delete the archive after extraction
            os.remove(file_path)
            print(f"Deleted: {file}")

    print("little pause...")
    time.sleep(4)

    zip_files = []
    year_data_folders = []

    # Check for nested archives again (second pass)

    for file in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file)
        if file.endswith((".zip", ".tar", ".rar", ".7z", ".gz", ".bz2", ".xz")):
            zip_files.append(file_path)
            new_folder_name = file.split(".")
            new_folder = os.path.join(input_folder, new_folder_name[0])
            os.makedirs(new_folder, exist_ok=True)
            ##
            year_data_folders.append(new_folder)

    paired_args = list(zip(zip_files, year_data_folders))

    #Unzip all years for that station at the same time:
    if len(zip_files) > 0:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(unzip_year, paired_args)

        print("Successfully unzipped the year files for station.")
    else:
        print("Issue with getting files ready for unzipping.")


# Function to load .raw file into env:
def load_raw(file_path):
    """Load data from a .raw file."""
    try:
        return np.loadtxt(file_path, delimiter=',', dtype=int)
    except ValueError:
        #logger.info(f"Skipping .raw file {os.path.basename(file_path)} due to load error.")
        return None
    

# Function to load .spt file into env:    
def load_spt(file_path):
    """Load spectral wave data from an .spt file into a dictionary."""
    
    try:
        df = pd.read_csv(file_path, skiprows=12, header=None, delimiter=',')
        df.columns = ['waveFrequency', 'waveEnergyDensity', 'waveMeanDirection', 'waveSpread', 'Skewness', 'Kurtosis']
        df = df[['waveFrequency', 'waveEnergyDensity', 'waveMeanDirection', 'waveSpread']]
        
        # Ensure all columns have exactly 64 values**
        for col in df.columns:
            if len(df[col]) != 64:
                df[col] = np.pad(df[col], (0, 64 - len(df[col])), constant_values=np.nan)

        return {col: df[col].values for col in df.columns}
    
    except Exception as e:
        #logger.info(f"Skipping .spt file {os.path.basename(file_path)}: {e}")
        return None    


def calculate_time(datetime_obj):
    """Calculate time array based on base time from datetime object."""
    
    # Calculate total seconds since the Unix epoch and convert to timedelta64[ns]
    total_seconds = (datetime_obj - pd.Timestamp('1990-01-01T00:00:00')).total_seconds()
    total_timedelta = np.timedelta64(int(total_seconds),'s')  # Convert to timedelta64[s] - total seconds between 1970,1,1 and the start of the sample

    seconds_from_epoch = total_timedelta + seconds_from_start
    datetime_array = np.datetime64('1990-01-01') + seconds_from_epoch  # convert to datetime
    numtime_array = (datetime_array.astype('datetime64[ns]') - np.datetime64('1990-01-01'))  # convert to timedelta64 (seconds since)

    return numtime_array.astype('datetime64[ns]')  # Ensure the output is of type datetime64[ns] !


def extract_datetime(file_path):
    """Extract datetime from filename (files contain no time information!)"""
    match = re.search(r'\d{4}-\d{2}-\d{2}T\d{2}h\d{2}', file_path)
    if match:
        datetime_str = match.group()
        return pd.Timestamp(datetime_str.replace('h', ':'))
    return None


def process_raw(raw_month_files, station_name):
    """Process raw files for a specific month, combining vertical dispalcement data, creating a time variable, and filling missing data with NaNs."""
    
    extracted_dates = [extract_datetime(f) for f in raw_month_files if extract_datetime(f)]
    if not extracted_dates:
        #logger.info("No valid raw files found in the provided list. Skipping...")
        return

    # Find unique (year, month) pairs
    unique_months = sorted(set((d.year, d.month) for d in extracted_dates))

    combined_data = np.zeros((0, 4))
    combined_time = np.empty((0, num_samples), dtype='datetime64[ns]')
    
    for year, month in unique_months:
        num_days = calendar.monthrange(year, month)[1]

        # Generate expected timestamps (1 file per 30 mins â†’ 48 files per day)
        expected_dates = [pd.Timestamp(year, month, day, hour, minute)
                          for day in range(1, num_days + 1)
                          for hour in range(0, 24)
                          for minute in (0, 30)]  # 30-minute intervals

        incomplete_files = 0
        missing_files = 0

        for expected_time in expected_dates:
            # Find the matching file for this expected timestamp
            file_match = next((f for f in raw_month_files if extract_datetime(f) == expected_time), None)

            if file_match:
                file_path = file_match
                current_data = load_raw(file_path)
                if current_data is None or current_data.ndim == 0 or len(current_data) != num_samples:
                    incomplete_files += 1
                    current_data = np.full((num_samples, 4), np.nan)
            else:
                missing_files += 1
                current_data = np.full((num_samples, 4), np.nan)

            # Add the time (calculated)
            time = calculate_time(expected_time).reshape(1, -1)
            combined_data = np.vstack((combined_data, current_data))
            combined_time = np.vstack((combined_time, time))


    # Return the processed data as an xarray dataset
    raw_ds = return_raw_ds(combined_data, combined_time.ravel(), f"{year}", station_name)
        
    logger.info(f"[.raw] Sucessfully processed {len(raw_month_files)} files.")  # Found {incomplete_files} incomplete, {missing_files} missing files
    return raw_ds
    
    
def return_raw_ds(combined_data, time, datetime_str, station_name):
    """Write the combined month data and time to a xarray dataset."""
    
    df = pd.DataFrame(combined_data, columns=['status', 'z_displacement', 'y_displacement', 'x_displacement'])
    disp = df['z_displacement'].values / 100  # Convert from cm to m

    ds = xr.Dataset(
        {
            'displacement': (('time',), disp),
        },
        coords={
            "time": time,
        })
    return ds

    
def process_spt(spt_file_list, station_name):
    """Process a given list of .spt files, ensuring 64 values per timestamp."""

    if not spt_file_list:
        #logger.info(f"No .spt files provided for station {station_name}. Skipping...")
        return None

    spectral_data_list = []
    time_list = []

    for spt_file in spt_file_list:
        spt_time = extract_datetime(spt_file)  # Extract timestamp from filename
        if not spt_time:
            #logger.info(f"Skipping file with unknown timestamp: {spt_file}")
            continue

        spectral_data = load_spt(spt_file)  # Load .spt file data
        if spectral_data:
            spectral_data_array = np.column_stack([spectral_data[key] for key in spectral_data])  
            spectral_data_list.append(spectral_data_array)
            
            # Convert timestamp to np.datetime64
            time_list.append(np.datetime64(spt_time))

    if not spectral_data_list:
        #logger.info("No valid spectral data found in provided .spt files.")
        return None

    # Stack into a NumPy array
    spectral_data_list = np.stack(spectral_data_list, axis=0)
    time_array = np.array(time_list).astype('datetime64[ns]')

    ds = return_spt_ds(spectral_data_list, time_array)
        
    logger.info(f"[.spt ] Sucessfully processed {len(spt_file_list)} files")
    return ds
                 
            
def return_spt_ds(spectral_data_list, time_list):
    """Generate and return an xarray dataset containing spectral data."""

    num_times, num_frequencies, num_vars = spectral_data_list.shape  # Ensure correct shape
    spectral_vars = {}

    # Calculate waveDp (peak wave direction at max energy density)
    waveMeanDirection = spectral_data_list[:, :, 2]
    waveDp_indices = np.argmax(spectral_data_list[:, :, 1], axis=1)  # Find index of max energy density for each time step
    waveDp = waveMeanDirection[np.arange(num_times), waveDp_indices]  # Extract corresponding mean direction

    # Define variable names
    var_names = ["waveEnergyDensity", "waveMeanDirection", "waveSpread", "waveDp"]

    # Populate spectral_vars dictionary
    for i, var_name in enumerate(var_names):
        if var_name == "waveDp":
            spectral_vars['waveDp'] = ('waveTime', waveDp)  # Store waveDp as a 1D array with time dim
        else:
            spectral_vars[var_name] = (('waveTime', 'waveFrequency'), spectral_data_list[:, :, i+1])

    # Create xarray Dataset
    ds = xr.Dataset(
        spectral_vars,
        coords={
            'waveTime': np.array(time_list),  # Use the provided time list
            'waveFrequency': np.arange(num_frequencies),  # Frequency indices (0-63)
        })
    return ds 


def process_raw_spt_files(raw_year_path, spt_year_path, nc_cco_path, station_name, attrs):
    """Process .raw and .spt files and write to netcdf file"""
    
    # Check for multiprocessing 
    pid = os.getpid()
    logger.info(f"Processing {raw_year_path} & {spt_year_path} on PID: {pid}")

    # Collect all .raw and .spt files (in each year folder) using list comprehensions
    raw_files = [os.path.join(root, file) for root, _, files in os.walk(raw_year_path) for file in files if file.endswith('.raw')]
    spt_files = [os.path.join(root, file) for root, _, files in os.walk(spt_year_path) for file in files if file.endswith('.spt')]
    
    # Extract timestamps from filenames
    raw_dates = {f: extract_datetime(f) for f in raw_files if extract_datetime(f) is not None}
    spt_dates = {f: extract_datetime(f) for f in spt_files if extract_datetime(f) is not None}
    
    # Get unique (year, month) pairs
    unique_months = sorted(set((dt.year, dt.month) for dt in raw_dates.values()) |
                           set((dt.year, dt.month) for dt in spt_dates.values()))
    
    # Loop over unique month files
    for year, month in unique_months:  
        raw_month_files = [f for f, dt in raw_dates.items() if dt.month == month]
        spt_month_files = [f for f, dt in spt_dates.items() if dt.month == month]
    
        raw_ds = process_raw(raw_month_files, station_name) if raw_month_files else None
        spt_ds = process_spt(spt_month_files, station_name) if spt_month_files else None
            
        # Write temp .pkl file
        if raw_ds and spt_ds:
            combined_ds = xr.merge([raw_ds, spt_ds], compat='override')
            combined_ds.attrs = {**attrs.attrs, **spt_ds.attrs}
        
            # Save the combined dataset as a temporary pickle file
            pkl_filename = os.path.join(nc_cco_path, f"{station_name}_{year}_{month:02d}.pkl")
        
            with open(pkl_filename, "wb") as pkl_file:
                pickle.dump(combined_ds, pkl_file)
            
            logger.info(f"Saved intermediate dataset as: {station_name}_{year}_{month:02d}.pkl")
        
            combined_ds.close()  # Free memory


def merge_pkl_into_netcdf(processed_data_path):
    """ Load all pickled datasets, merge them, and write to NetCDF """
    
    # Get all pickle files
    pkl_files = [file for file in os.listdir(processed_data_path) if file.endswith('.pkl')]
    
    # Merge all files in sequential order with error handling for invalid filenames
    pkl_files = sorted(
        [os.path.join(processed_data_path, file) for file in pkl_files],
        key=lambda f: (
            # Ensure the year part (f.split('_')[1]) is numeric
            int(f.split('_')[1]) if f.split('_')[1].isdigit() else 0,  
            # Ensure the month part (f.split('_')[-1].split('.')[0]) is numeric
            int(f.split('_')[-1].split('.')[0]) if f.split('_')[-1].split('.')[0].isdigit() else 0  
        )
    )


    datasets = []
    for pkl_file in pkl_files:
        with open(pkl_file, "rb") as file:
            datasets.append(pickle.load(file))
    

    final_ds = xr.merge(datasets, compat='override')

    nc_filename = os.path.join(processed_data_path, f"{os.path.basename(processed_data_path)}.nc")  # Create nc filename
   
    final_ds.to_netcdf(nc_filename)  # Write to NetCDF (efficient batch write)

    logger.info(f"Saved final merged NetCDF as: {os.path.basename(processed_data_path)}")
    
    # Cleanup
    final_ds.close()
    for pkl_file in pkl_files:
        os.remove(pkl_file)  # Delete temp files


def paralell_process_files_by_station(raw_station_path, spt_station_path, nc_cco_path, station_name, attrs):
    """Take both folders containing the yearly [.raw & .spt] data and processes all years in paralell"""
    
    logger.info(f"Currently in file: {station_name}")
    raw_year_folder = [f for f in os.listdir(os.path.join(raw_station_path))] # contains all yearly 1Hz data for one station - looping through each station in the main program
    spt_year_folder = [f for f in os.listdir(os.path.join(spt_station_path))]
   
    # Run the files into the process_files_by_month function for processing
    raw_year_path = []
    spt_year_path = []
    
    # Get the full paths for all year folders
    for j in range(len(raw_year_folder)):
        raw_year_path.append(os.path.join(raw_station_path, raw_year_folder[j])) 
        spt_year_path.append(os.path.join(spt_station_path, spt_year_folder[j]))

    number_of_years = len(raw_year_folder)
    paired_args = list(zip(raw_year_path, spt_year_path, repeat(nc_cco_path, number_of_years), repeat(station_name, number_of_years), repeat(attrs, number_of_years)))

    if number_of_years > 0: 
        print(f"Starting multiprocessing {number_of_years} years...")
        logger.info(f"Starting multiprocessing {number_of_years} years...")
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            multiprocessing.log_to_stderr(logging.INFO)
            pool.starmap(process_raw_spt_files, paired_args) # Passing .raw and .spt path into function
            
        logger.info("Finished multiprocessing")
        time.sleep(2)
    else:
        logger.info("No year files for the station to process.")


def setup_file_logger(logfile):
    """Set up basic file logger (copied from .logs FOWD).""" 
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=logfile,
        filemode='w'
    )
    logging.captureWarnings(True)

    # silence third-party loggers
    logging.getLogger('filelock').setLevel('CRITICAL')
    logging.getLogger('multiprocessing').setLevel(logging.WARNING)
    
    

# Main program: 
# Global variables:
sample_rate = 1.28  # Datawell Waverider output sample rate (Hz) 
record_length = 30 * 60  # Output files are 30 minutes in length
num_samples = int(sample_rate * record_length)

# Create SECONDS_FROM_START with nanosecond precision
seconds_from_start = ((np.linspace(0, num_samples - 1, num_samples) / sample_rate) * 1e9).astype('timedelta64[ns]')

# Set CWD
os.chdir(r"/scratch/scw2368/spt_download_test/1_station_test")  # 

# Set up multiprocessing logging
logfile = os.path.join(
    os.getcwd(),
    'pre-process.log'  # {datetime.datetime.today():%Y%m%dT%H%M%S}
)
setup_file_logger(logfile)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    
    # Load station data
    station_metadata_path = os.path.join(os.getcwd(), "station_metadata.csv")
    station_metadata = pd.read_csv(station_metadata_path)

    # Loop through stations
    for i in range(1):  # Adjust for all stations if needed

        # Extract station name
        station_name = station_metadata["station_name"][i]

        # # Create station directory
        unprocessed_data_path = os.path.join(os.getcwd(),'unprocessed_data', station_metadata["station_name"][i])
        os.makedirs(unprocessed_data_path, exist_ok=True)

        # Download station data from the CCO website - looping through both data types (1Hz and Wave Spectra)
        for data_type, tab_name, button_name in zip(["1Hz", "Wave_Spectra"], ["1Hz%20Data", "Wave%20Spectra"], ["selalyearsraw", "selalyears"]):
            
            # Set the download folder for each type
            data_dir = os.path.join(unprocessed_data_path, f"{data_type}")
            os.makedirs(data_dir, exist_ok=True)

            # Configure Edge options
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option("prefs", {
                "download.default_directory": data_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True
            })

            service = Service("/scratch/scw2368/edge_stuff/edge/edge_browser/opt/microsoft/msedge/msedgedriver")
            driver = webdriver.Edge(service=service, options=options)

            # Generate the correct URL
            web_url = f"https://coastalmonitoring.org/realtimedata/?chart={station_metadata['station_code'][i]}&tab={tab_name}&disp_option=2"
            driver.get(web_url)

            try:
                wait = WebDriverWait(driver, 10)

                # Click checkbox to select all years
                checkbox = wait.until(EC.presence_of_element_located((By.ID, button_name)))
                driver.execute_script("arguments[0].scrollIntoView();", checkbox)
                driver.execute_script("arguments[0].click();", checkbox)

                # Click download button
                download_button = wait.until(EC.element_to_be_clickable((By.NAME, "download_yearly_files")))
                driver.execute_script("arguments[0].scrollIntoView();", download_button)
                driver.execute_script("arguments[0].click();", download_button)

                # Wait for download to complete
                wait_for_download(data_dir)
                print(f"Downloaded {data_type} data into: {data_dir}")

            finally:
                driver.quit()

            # Unzip the downloaded files
            print(f"Unzipping {data_type} data...")
            time.sleep(2)
            unzip_nested_archives(data_dir)
            print(f"{station_name} {data_type} data unzipping complete.")

            time.sleep(2)
        
        # Package attributes 
        attrs = xr.Dataset(attrs = {
            "longitude": station_metadata["lon"][i],
            "latitude": station_metadata["lat"][i],
            "water_depth":  station_metadata["depth"][i],
            "sampling_rate": sample_rate,
            "station_name": station_name,
            "spring_tidal_range": station_metadata["spr_tidal_rng"][i],
            "storm_threshold": station_metadata["storm_thresh"][i],
            "region": station_metadata["region"][i],
            "deployment_date": station_metadata["deployment_date"][i],
        })
        
        
        # Create folders to save the .nc files and also specify the .nc outputs:
        processed_data_path = os.path.join(os.getcwd(), "processed_data",station_name)
        os.makedirs(processed_data_path, exist_ok=True)
             
        # Specify unprocessed data paths
        raw_dataset_path = os.path.join(unprocessed_data_path,'1Hz')
        spt_dataset_path = os.path.join(unprocessed_data_path,'Wave_Spectra')

        # Call main processing function
        paralell_process_files_by_station(raw_dataset_path, spt_dataset_path, processed_data_path, station_name, attrs)
        print(f"Processing complete for: {station_name}")
        
        merge_pkl_into_netcdf(processed_data_path)
        
        # Delete data files after processing
        delete_folder_contents(raw_dataset_path)
        delete_folder_contents(spt_dataset_path)
        print("Deleted .raw files for station, now combining into netcdf...")
        


