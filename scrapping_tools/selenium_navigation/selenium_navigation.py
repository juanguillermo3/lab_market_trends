import os
import logging
import time
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
                 self.driver_path 
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
    def navigate_to(self, url, timeout=10, title_check=None):
        """
        Navigate to a given URL and optionally wait for a page title or a specific condition.

        Args:
            url (str): The URL to navigate to.
            timeout (int): The maximum time to wait for the page to load or a condition to be met (default is 10 seconds).
            title_check (str, optional): The expected page title to confirm successful navigation.
        """
        try:
            self.logger.info(f"Attempting to navigate to URL: {url}")
            self.driver.get(url)
            
            # Optionally wait for the page title or a specific condition
            if title_check:
                self.logger.info(f"Waiting up to {timeout} seconds for the page title to be: '{title_check}'")
                WebDriverWait(self.driver, timeout).until(
                    EC.title_contains(title_check)
                )
                self.logger.info(f"Navigation successful: Page title contains '{title_check}'")
            else:
                # Wait until some basic page loading finishes
                WebDriverWait(self.driver, timeout).until(
                    lambda d: d.execute_script('return document.readyState') == 'complete'
                )
                self.logger.info("Page loaded successfully (document.readyState is 'complete')")
        
        except Exception as e:
            self.logger.error(f"Failed to navigate to URL: {url}. Error: {e}")
            raise

    #
    # (1)
    #


    def heartbeat(self, xpath, max_wait=10):
        try:
            # Log the start of the waiting process
            self.logger.info(f"Waiting up to {max_wait} seconds for element to appear using XPath: {xpath}")
            
            start_time = time.time()  # Record the start time
            
            # Wait for the element to appear
            WebDriverWait(self.driver, max_wait).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            self.logger.info(f"Element found for XPath: {xpath} in {elapsed_time:.2f} seconds")
            return True
        
        except Exception as e:
            # Log failure if element is not found within the given time
            elapsed_time = time.time() - start_time
            self.logger.warning(f"Element not found for XPath: {xpath} after {elapsed_time:.2f} seconds. Error: {e}")
            return False


    #
    # (2)
    #
    def find_elements_by_xpath(self, xpath, wait=True, max_wait=10):
        try:
            start_time = time.time()  # Record the start time
            
            if wait:
                self.logger.info(f"Waiting up to {max_wait} seconds for elements to appear using XPath: {xpath}")
                WebDriverWait(self.driver, max_wait).until(
                    EC.presence_of_all_elements_located((By.XPATH, xpath))
                )
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                self.logger.info(f"Elements found for XPath: {xpath} in {elapsed_time:.2f} seconds")
            
            elements = self.driver.find_elements(By.XPATH, xpath)
            
            # Log the number of elements found
            if len(elements) == 0:
                self.logger.warning(f"No elements found for XPath: {xpath}")
            else:
                self.logger.debug(f"XPath used: {xpath}")
                self.logger.info(f"Number of elements found: {len(elements)}")
            
            return elements
        
        except Exception as e:
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            self.logger.error(f"An error occurred while finding elements with XPath '{xpath}' after {elapsed_time:.2f} seconds: {e}")
            return []


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
