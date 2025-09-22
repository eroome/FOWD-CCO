"""
download_CCO.py

Download entire wave buoy data archive from https://coastalmonitoring.org/

Inputs:
    - 'station_metadata.csv' metadata file containing station names, latitude, longitude, etc. (required)

Requirements:
    - Edge driver

"""

import os
import pandas as pd
import time
import multiprocessing
import logging 
import zipfile

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait



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
                        
    else:
        logger.info("Parent folder does not exist")


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

    logger.info("Download timeout or file not found.")
    return False


def extract_zip(file_path, output_folder):
    """Extract a .zip archive quietly."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        os.remove(file_path)
    except Exception as e:
        logger.warning(f"Failed to extract {file_path}: {e}")


def unzip_year(zip_file, year_folder):
    extract_zip(zip_file, year_folder)


def unzip_nested_archives(input_folder):
    """Extracts .zip archives in a folder, including nested ones."""
    # First pass: extract directly into input folder
    for file in os.listdir(input_folder):
        if file.endswith(".zip"):
            file_path = os.path.join(input_folder, file)
            logger.info(f"Extracting top-level: {file}")
            extract_zip(file_path, input_folder)

    time.sleep(2)

    # Second pass: nested .zip files go into named folders
    zip_files = []
    year_data_folders = []

    for file in os.listdir(input_folder):
        if file.endswith(".zip"):
            file_path = os.path.join(input_folder, file)
            folder_name = os.path.splitext(file)[0]
            output_path = os.path.join(input_folder, folder_name)
            os.makedirs(output_path, exist_ok=True)
            zip_files.append(file_path)
            year_data_folders.append(output_path)

    if zip_files: 
        logger.info(f"Extracting {len(year_data_folders)} sub-folders") 
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.starmap(unzip_year, zip(zip_files, year_data_folders))
        

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

    # Silence third-party loggers
    logging.getLogger('filelock').setLevel('CRITICAL')
    logging.getLogger('multiprocessing').setLevel(logging.WARNING)
    

# Set up multiprocessing logging
logfile = os.path.join(os.getcwd(), f"download_cco_data_archive_{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}.log") # {datetime.datetime.today():%Y%m%dT%H%M%S} 
setup_file_logger(logfile)
logger = logging.getLogger(__name__)

# Main program: 
if __name__ == '__main__':
    
    # Load station data
    station_metadata_path = os.path.join(os.getcwd(), "station_metadata.csv")
    station_metadata = pd.read_csv(station_metadata_path)

    # Loop through stations
    for i in range(33,len(station_metadata)):

        # Extract station name
        station_name = station_metadata["station_name"][i]
        logger.info(f"Started downloading: {station_name}")

        # # Create station directory
        unprocessed_data_path = os.path.join(os.getcwd(),'unprocessed_data', station_metadata["station_name"][i])
        os.makedirs(unprocessed_data_path, exist_ok=True)

        # Download station data from the CCO website - looping through both data types (1Hz and Wave Spectra)
        for data_type, tab_name, check_button, dl_button in zip(["1Hz", "Wave_Spectra", 'CCO_QC'], ["1Hz%20Data", "Wave%20Spectra",'download'], 
        ["selalyearsraw", "selalyears", "chkboxwarningwaves"], ["download_yearly_files", "download_yearly_files", "qcsubmitwaves"]):
            
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
                
                if data_type == "CCO_QC":
                    
                    tick_ogl = wait.until(EC.element_to_be_clickable((By.NAME, 'agree_ogl')))
                    driver.execute_script("arguments[0].scrollIntoView();", tick_ogl)
                    driver.execute_script("arguments[0].click();", tick_ogl)
                    
                    # Wait for any checkboxes to appear
                    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'input.filelistwaves[type="checkbox"]')))          
                    checkboxes = driver.find_elements(By.CSS_SELECTOR, 'input.filelistwaves[type="checkbox"]')
                    
                    for box in checkboxes:
                        checkbox_id = box.get_attribute("id")
                        
                        if not box.is_selected():
                            # Scroll into view and wait
                            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", box)
                            time.sleep(0.2)  # tiny delay helps headless mode
                            
                            # Use JS click to avoid intercept issues
                            driver.execute_script("arguments[0].click();", box)              

                else:
                    # Click checkbox to select all years
                    checkbox = wait.until(EC.presence_of_element_located((By.ID, check_button)))
                    driver.execute_script("arguments[0].scrollIntoView();", checkbox)
                    driver.execute_script("arguments[0].click();", checkbox)

                # Click downlaod button to get data         
                if data_type == "CCO_QC":
                    download_button = wait.until(EC.element_to_be_clickable((By.ID, dl_button)))
                else:
                    download_button = wait.until(EC.element_to_be_clickable((By.NAME, dl_button)))
                    
                driver.execute_script("arguments[0].scrollIntoView();", download_button)
                driver.execute_script("arguments[0].click();", download_button)
                     
                  
                # Wait for download to complete
                wait_for_download(data_dir)
                logger.info(f"Downloaded {data_type} data ({station_name})")

            finally:
                driver.quit()
                
                
            # Unzip the downloaded files
            time.sleep(2)
            unzip_nested_archives(data_dir)
            logger.info(f"Unzipped {data_type} data ({station_name})")

            time.sleep(2)