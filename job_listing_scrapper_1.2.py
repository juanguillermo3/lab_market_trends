"""
project: lab_market_trends
title: Jobs Vacancies scrapper
description: This modules performs scrapping of job listings
image: job_listings_scrapper.png
Author: Juan Guillermo
Include in Portfolio: True
"""

#
# I will use a dotenv approach to configure the scrapper job. This approach
# allow me to protect sensitive information such as the url source of the data. It
# also helps to centralize some configuration defining the temporal scope of the
# scrapping, such as the target initial starting date the scrapper is positioned 
# at, and numebr of days to scrap retrospectivelly. Finally, will use the pattern
# of and environmental variable to discrimiante production vs development environments.
#
#



import os
#
try:
    from dotenv import load_dotenv
except:
    os.system("pip install python-dotenv")
    from dotenv import load_dotenv
#
from datetime import datetime, timedelta

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
SCRAPPING_URL_TARGET = os.getenv("SCRAPPING_URL_TARGET")
ENVIRONMENT = os.getenv("ENVIRONMENT")
SINK_FOLDER= os.getenv("SINK_FOLDER") #'~/lab_market_trends/scapping_lab_market/'
NUMBER_OF_DAYS_TO_SCRAP = int(os.getenv("NUMBER_OF_DAYS_TO_SCRAP"))
SELENIUM_DRIVER_PATH = os.getenv("SELENIUM_DRIVER_PATH")   #"/usr/bin/chromedriver"
NUMBER_OF_JOBS_TO_SCRAP= None

SCRAPPING_URL_TARGET 
ENVIRONMENT
SINK_FOLDER
SELENIUM_DRIVER_PATH

# Set the waiting time based on the environment
if ENVIRONMENT == "production":
    WAIT_LEN = 0.2  # 0.2 seconds otherwise
else:
    WAIT_LEN = 3  # 3 seconds in production
#
WAIT_LEN

#
# The selenium class streamlines and abstracts around many tipicall task of
# programatic browser navigation using Selenium. Its build around the idea


import os
import logging

#
try:
    import selenium
except:
    os.system("pip install selenium")
    import selenium

#
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains


class SeleniumNavigation:
    
    #
    # 0.
    #
    def __init__(self, driver_path, download_dir=None, browser_type='chrome', headless=False):
        self.driver_path = driver_path
        self.download_dir = download_dir
        self.browser_type = browser_type
        self.headless = headless
        self.driver = None
        
        # Set up the logger for this class
        self.logger = logging.getLogger(__name__)  # This ensures that only logs from this class/module are captured
        self.logger.setLevel(logging.DEBUG)  # You can use DEBUG, INFO, etc., as needed
        
        # Suppress logging from external libraries like Selenium
        logging.getLogger("selenium").setLevel(logging.CRITICAL)  # Adjust this level as needed
        logging.getLogger("urllib3").setLevel(logging.CRITICAL)
        logging.getLogger("requests").setLevel(logging.CRITICAL)
        
        
    #
    # 1. context handling capabilities
    #
    def __enter__(self):
        self._initialize_driver()
        return self
    #
    def _initialize_driver(self):
        options = Options()
        if self.headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")

        if self.browser_type == 'chrome':
            # Configure Chrome specific options for downloading files
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False,  # Disable safe browsing
                "safebrowsing.disable_download_protection": True,  # Disable download protection
                "profile.default_content_setting_values.automatic_downloads": 1  # Allow automatic downloads
            }
            options.add_experimental_option("prefs", prefs)
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-popup-blocking")

            service = Service(
                SELENIUM_DRIVER_PATH
                )
            
            self.driver = webdriver.Chrome(service=service, options=options)

        return self.driver

    def __exit__(self, exc_type, exc_value, traceback):
        self.close_browser()
    #
    def close_browser(self):
        if self.driver:
            self.driver.quit()

    #
    # 2. Within context capabilities
    #
    
    #
    # (0)
    #
    def navigate_to(self, url):
        self.driver.get(url)
    #
    # (1)
    #
    def heartbeat(self, xpath, max_wait=10):
        try:
            WebDriverWait(self.driver, max_wait).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            return True
        except:
            return False
    #
    # (2)
    #
    def find_elements_by_xpath(self, xpath, wait=True, max_wait=10):
        try:
            if wait:
                WebDriverWait(self.driver, max_wait).until(
                    EC.presence_of_all_elements_located((By.XPATH, xpath))
                )
            elements = self.driver.find_elements(By.XPATH, xpath)
            
            # Log the number of elements found
            self.logger.debug(f"XPath used: {xpath}")
            if len(elements) == 0:
                self.logger.warning(f"No elements found for XPath: {xpath}")
            else:
                self.logger.debug(f"Number of elements found: {len(elements)}")
            
            return elements
        
        except Exception as e:
            self.logger.error(f"An error occurred while finding elements with XPath '{xpath}': {e}")
            return []
    #
    # (3)
    #
    def find_child_elements(self, parent_element, xpath):
        try:
            child_elements = parent_element.find_elements(By.XPATH, xpath)
            return child_elements
        except Exception as e:
            print(f"An error occurred while finding child elements: {e}")
            return []
    #
    # (4)
    #
    def move_and_click(self, xpath, max_wait=10):
        try:
            # Find elements by the given XPath
            elements = self.find_elements_by_xpath(xpath, max_wait=max_wait)
            
            # Ensure only one element is matched
            if len(elements) != 1:
                raise ValueError(f"Multiple elements matched for XPath: {xpath}. Please provide a more precise XPath.")
            
            # Move to the element and click it
            element = elements[0]
            actions = ActionChains(self.driver)
            actions.move_to_element(element).perform()
            element.click()
            
            self.logger.debug(f"Clicked element with XPath: {xpath}")
        
        except ValueError as e:
            self.logger.error(f"Error: {e}")
            raise

        
# Usage:
#driver_path = '/path/to/your/chromedriver'
#with SeleniumNavigation(driver_path, headless=True, download_dir=THIS_ETL_HOME) as navigator:
#    navigator.navigate_to('https://example.com')
#    if navigator.heartbeat('//h1'):
#        print("Access to the website is successful, and the h1 element is present.")
#    else:
#        print("Access to the website failed, or the h1 element is not present.")


#
# JobsSiteNavigation extends the SeleniumNavigation class, providing some insite navigation
#

import re
from selenium.webdriver.common.keys import Keys  # Ensure this import is present
from dateutil.parser import parse as parse_date
from datetime import datetime, timedelta

class JobsSiteNavigation(SeleniumNavigation):
    def navigate_to_page(self, page_number, max_wait=10):
        """Navigate to a specific page by entering the page number."""
        input_element = self.find_elements_by_xpath('//p[@class="pagination-input-container false"]//input[@placeholder]', max_wait=max_wait)
        
        # Ensure there's a valid input element
        if input_element:
            input_element[0].clear()
            input_element[0].send_keys(str(page_number))
            input_element[0].send_keys(Keys.RETURN)
            print("Navigated to page:", page_number)
        else:
            # If the element is not found, raise an error and interrupt the flow
            raise ValueError(f"Input element for pagination not found.")

    def fetch_listings(self, max_wait=10):
        """Fetch job listings from the current page."""
        try:
            results_xpath = '//div[@class="results-card-container"]'
            if self.heartbeat(results_xpath, max_wait):
                elements = self.find_elements_by_xpath(results_xpath, max_wait=max_wait)
                listings = [element.text for element in elements]
                self.logger.info(f"Fetched {len(listings)} listings.")
                return listings
            else:
                self.logger.warning("No listings found on the current page.")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching listings: {e}")
            return []
    

    def fetch_last_scraped_date(self, max_wait=10):
        """Fetch the last scraped date from the most recent listing."""
        try:
            listings = self.fetch_listings(max_wait=max_wait)
            
            if listings:
                # Assume the last listing has a string with "Fecha de Publicación" and the date
                last_listing_text = listings[-1]  # Get the last element from listings

                # Search for "Fecha de Publicación" and extract the date from the last listing
                last_date_text = last_listing_text[re.search("Fecha de Publicación", last_listing_text).span()[0]:]

                # Use regex to extract the date and parse it
                date_string = "/".join(re.findall(r"[0-9]+", last_date_text))
                last_date_scrapped = parse_date(date_string)
                
                self.logger.info(f"Last date scraped: {last_date_scrapped}")
                return last_date_scrapped
            else:
                self.logger.warning("No listings available to extract a date.")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching last scraped date: {e}")
            return None
            

    def navigate_to_date(self, target_date, max_steps=120, pages_per_step=100, max_wait=10):
        """
        Navigate to a page containing listings from the desired date.

        Args:
            target_date (datetime): The target date to navigate to.
            max_steps (int): Maximum number of steps to try.
            pages_per_step (int): Number of pages to move forward per step.
            max_wait (int): Maximum wait time for page elements.

        Raises:
            ValueError: If the provided date is after the current position.
            RuntimeError: If navigation fails to find the desired date or surpasses it.
        """
        # Ensure the target_date is a datetime object
        if not isinstance(target_date, datetime):
            raise ValueError("The target_date must be a datetime object.")

        # Truncate the target_date to the day (timezone-unaware)
        target_date = target_date.replace(tzinfo=None).date()

        # Fetch the current system date (timezone-unaware)
        current_date = datetime.now().replace(tzinfo=None).date()  # Get system date and make it naive

        print("currently at:", current_date)

        # Check if we are already at the target date
        if current_date == target_date:
            self.logger.info(f"Already at the target date: {current_date}. No navigation needed.")
            return

        # Fetch the last scraped date from the scraper
        current_date_scraper = self.fetch_last_scraped_date(max_wait=max_wait)

        if current_date_scraper is None:
            raise RuntimeError("Unable to determine the current page's date.")

        # Truncate the current_date_scraper to the day (timezone-unaware)
        current_date_scraper = current_date_scraper.replace(tzinfo=None).date()

        # Validate the input date
        if target_date > current_date_scraper:
            raise ValueError("The provided date is after the current page's date. Cannot navigate forward.")

        for step in range(max_steps):
            # Check the date on the current page
            current_date_scraper = self.fetch_last_scraped_date(max_wait=max_wait)

            print("currently at:", current_date_scraper)

            if current_date_scraper is None:
                raise RuntimeError(f"Unable to fetch the date during navigation at step {step + 1}.")

            # Truncate the current_date_scraper to the day (timezone-unaware)
            current_date_scraper = current_date_scraper.replace(tzinfo=None).date()

            if current_date_scraper == target_date:
                self.logger.info(f"Reached the target date: {current_date_scraper}.")
                return

            if current_date_scraper < target_date:
                # Raise an error if we pass the desired date
                raise RuntimeError(
                    f"Navigation error: Surpassed the target date {target_date} at page {step + 1} with date {current_date_scraper}."
                )

            # Determine the current page number
            input_element = self.find_elements_by_xpath(
                '//p[@class="pagination-input-container false"]//input[@placeholder]',
                max_wait=max_wait
            )
            if not input_element or not input_element[0].get_attribute("placeholder").isdigit():
                raise RuntimeError("Unable to determine the current page number.")
            
            current_page = int(input_element[0].get_attribute("placeholder"))
            next_page = current_page + pages_per_step

            # Navigate to the next page
            self.navigate_to_page(next_page, max_wait=max_wait)
            self.logger.info(f"Moved to page {next_page} (step {step + 1}).")

        # If we exhaust the steps without finding the date
        raise RuntimeError(f"Could not find the target date {target_date} within {max_steps} steps.")


import os
import re
import time
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from selenium.common.exceptions import NoSuchElementException, WebDriverException
import os
import gzip
from datetime import datetime
from airflow.exceptions import AirflowSkipException


# Function to calculate last Friday's date
def get_last_friday(reference_date=None):
    if reference_date is None:
        reference_date = datetime.now()
    offset = (reference_date.weekday() - 4) % 7  # 4 corresponds to Friday
    return reference_date - timedelta(days=offset)
#
get_last_friday()
        
# 1. Heartbeat Target URL
def heartbeat_target_url():
    with JobsSiteNavigation(driver_path=SELENIUM_DRIVER_PATH) as navigator:
        
        navigator.navigate_to(SCRAPPING_URL_TARGET)

        if navigator.heartbeat('//*[text()="Resultados"]'):

            # Accumulate data from pages 1, 300, and 500
            all_listings = []
            for page in [1, 300]:
                navigator.navigate_to_page(page, max_wait=30)
                listings = navigator.fetch_listings(max_wait=30)
                all_listings.extend(listings)
                print(f"Last scraped date on page {page}: {navigator.fetch_last_scraped_date()}")
                
            # Report the number of unique listings
            unique_listings = set(all_listings)
            print(f"Number of unique listings found: {len(unique_listings)}")

            # Reposition at the first page
            navigator.navigate_to_page(1, max_wait=30)
            print("Repositioned to the first page.")

            # Calculate target date (10 days ago)
            target_date = get_last_friday()

            # Move to the target date
            try:
                navigator.navigate_to_date(target_date, max_steps=120, pages_per_step=70, max_wait=30)
                print(f"Successfully moved to listings for the target date: {target_date.strftime('%Y-%m-%d')}")
            except RuntimeError as e:
                print(f"Error while navigating to target date: {e}")


# 2. Scrape Listings
def scrape_listings(
    number_of_days_to_scrap, 
    number_of_jobs_to_scrap,
    wait_len, 
    **kwargs
    ):
    #
    # (1) The configuration of this task defines the temporal scope of the scrapping process.
    # The gist of it is overriding the default configuration (scrapping from today up until
    # a number of days in the past), by some runtime configuration, allowing for different types
    # of scrapping (current vs backfilling workloads).
    #

    # Default parameters
    default_params = {
        'number_of_days_to_scrap': number_of_days_to_scrap, # sys default is 3
        'number_of_jobs_to_scrap': number_of_jobs_to_scrap,  # Default to None
        'execution_date': kwargs.get('execution_date')
    }

    # Merge defaults with runtime configuration
    dag_conf = kwargs.get('dag_run').conf 
    params = {**default_params, **dag_conf}
    
    number_of_days_to_scrap = params['number_of_days_to_scrap']
    number_of_jobs_to_scrap = params['number_of_jobs_to_scrap']
    target_date = params['execution_date']
    
    # Logging for debugging (optional)
    print(f"Scraping Parameters: days={number_of_days_to_scrap}, jobs={number_of_jobs_to_scrap}, target_date={target_date}")


    #
    # (2) Following the configuration step, the navigator is placed on the page where
    #

    # Helper function to parse date from string in dd/mm/yyyy format
    #def parse_date(date_string):
    #    return datetime.strptime(date_string, "%d/%m/%Y")

    H = []
    #WAIT_LEN=.1
    
    
    with JobsSiteNavigation(driver_path=SELENIUM_DRIVER_PATH) as navigator:
        #
        # (0)
        #
        navigator.navigate_to(SCRAPPING_URL_TARGET)
        #
        # (1)
        #
        if navigator.heartbeat('//*[text()="Resultados"]'):
            

            print("Positioning navigator at target date:", target_date)
            navigator.navigate_to_date(target_date, max_steps=120, pages_per_step=70, max_wait=40)
            print(f"Successfully moved to listings for the target date: {target_date.strftime('%Y-%m-%d')}")    
                
            # Optionally handle closing ads
            # ad_close_button = navigator.find_elements_by_xpath('//*[text()="Cerrar"]')[0]
            # ad_close_button.click()
            time.sleep(wait_len)

            # Handle total count fallback
            total_count = None
            try:
                total_count_element = navigator.find_elements_by_xpath(
                    "//*[text()='Número total de ofertas de empleo encontradas.']/following-sibling::*"
                )[0]
                total_count = float(total_count_element.text.replace(",", ""))
                print("Total job postings available:", total_count)
            except (IndexError, ValueError, AttributeError):
                print("Total count unavailable; proceeding without total count information.")

            # Correct for the navigation state before entering the while loop
            #def position(navigation):
            #    # Re-navigate or adjust navigation state if needed
            #    if not navigation.heartbeat('//div[@class="results-card-container"]'):
            #        navigation.navigate_to(SCRAPPING_URL_TARGET)
            #    time.sleep(WAIT_LEN)
            #    return navigation
            #
            #navigator = position(navigator)
            TARGET_REACHED = False
            while not TARGET_REACHED:
                try:
                    # Scrape the current page
                    H_ = [n.text for n in navigator.find_elements_by_xpath('//div[@class="results-card-container"]', max_wait=45)]
                    H.extend(H_)
                    if total_count:
                        print(f"Scraped {len(H)} entries ({round((len(H)/total_count)*100, 3)}%)")
                    else:
                        print(f"Scraped {len(H)} entries (progress percentage unknown).")

                    # Check if number of jobs target is reached
                    if number_of_jobs_to_scrap is not None and len(H) >= number_of_jobs_to_scrap:
                        TARGET_REACHED = True
                        print("Job count target reached.")
                        break

                    # Parse the last scraped date
                    last_date_text = H[-1][re.search("Fecha de Publicación", H[-1]).span()[0]:]
                    last_date_scrapped = parse_date("/".join(re.findall("[0-9]+", last_date_text)))
                    print("Last date scraped:", last_date_scrapped)

                    # Check if the date target is reached
                    days_diff = (target_date.replace(tzinfo=None).date() - last_date_scrapped.replace(tzinfo=None).date()).days
                    if days_diff > number_of_days_to_scrap:
                        TARGET_REACHED = True
                        print("Scraping goal reached.")
                        break

                    # Navigate to the next page
                    navigator.move_and_click('//button[@title="Página Siguiente"][1]', max_wait=45)
                    time.sleep(wait_len)
                except (NoSuchElementException, WebDriverException) as e:
                    print(f"Error navigating or scraping: {e}")
                    break  # Exit the loop and retry
                
    # Push scraped data to XCom
    kwargs['task_instance'].xcom_push(key='vacancies', value=H)

# 3. Export to Disk
def export_to_disk(folder, **kwargs):
    
    
    # Default parameters
    default_params = {
        'folder': folder,
    }
    
    # Fetch runtime parameters from dag_run.conf
    dag_conf = kwargs.get('dag_run').conf 
  
    
    # Merge defaults with runtime configuration
    params = {**default_params, **dag_conf}
    folder = params['folder']
    
    # Retrieve scraped data from XCom
    data = kwargs['task_instance'].xcom_pull(task_ids='scrape_listings', key='vacancies')
    
    if not data:
        raise AirflowSkipException("No data found in XCom to export.")
    
    if len(data)<10000: # should be corrected in num jobs asked is less than this
        raise AirflowSkipException("Insufficient data found for this run.")
    
    # Process the data to save each vacancy on a single line
    export_folder = os.path.expanduser(folder)
    
    # Ensure the folder exists
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    
    # Get the current date for filename
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Filename with date appended
    filename = f"vacancies_{current_date}.txt.gz"
    
    # Full file path
    file_path = os.path.join(export_folder, filename)
    
    # Write the cleaned data to the compressed text file
    with gzip.open(file_path, 'wt', encoding='utf-8') as f:
        for vacancy in data:
            # Clean up each vacancy by replacing internal newlines and carriage returns
            cleaned_vacancy = vacancy.replace('\n', ' ').replace('\r', ' ')
            # Write each cleaned vacancy to the file, each on a new line
            f.write(cleaned_vacancy + '\n')
    
    # Log the file path for debugging
    print(f"Data exported to: {file_path}")
    
    
#
# DAG definition
#

# Constants and default arguments
from datetime import datetime, timedelta
from airflow import DAG

DEFAULT_ARGS = {
    'owner': 'juan guillermo',
    'start_date': datetime(2024, 11, 15),  # Start backfilling from November 15, 2024
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'depends_on_past': False,  # Tasks do not depend on previous runs
}

with DAG(
    'job_listing_scraper_1.2',
    default_args=DEFAULT_ARGS,
    description='A DAG to scrape job listings periodically',
    schedule_interval='0 16 * * 0,3',  # Run at 2 PM on Sundays and Wednesdays
    catchup=True,  # Enable backfilling
    end_date=datetime(2025, 1, 15),  # End date for backfilling
)  as dag:

    # Task to perform a heartbeat check
    heartbeat_task = PythonOperator(
        task_id='heartbeat_target_url',
        python_callable=heartbeat_target_url,
        dag=dag,
        #provide_context=True
    )

    # Task to scrape job listings
    scrape_task = PythonOperator(
        task_id='scrape_listings',
        python_callable=scrape_listings,
        op_kwargs={
            'number_of_days_to_scrap': NUMBER_OF_DAYS_TO_SCRAP,
            'number_of_jobs_to_scrap': NUMBER_OF_JOBS_TO_SCRAP,  # Default to None
            "wait_len": WAIT_LEN
            },
        dag=dag,
        provide_context=True
    )

    # Task to export scraped data to disk
    export_task = PythonOperator(
        task_id='export_to_disk',
        python_callable=export_to_disk,
        op_kwargs={
            'folder': SINK_FOLDER,  # Folder path for saving the file
        },
        dag=dag,
        provide_context=True
    )

    # Task dependencies
    heartbeat_task >> scrape_task >> export_task



