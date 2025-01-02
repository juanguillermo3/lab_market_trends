import os
import json
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dagrun_operator import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Get the registry folder path, start date, and timeout from the environment variables
MANAGER_START_DATE = os.getenv("MANAGER_START_DATE")
MANAGER_REGISTRY_FOLDER = os.getenv("MANAGER_REGISTRY_FOLDER")
MANAGER_TIMEOUT_IN_HOURS = int(os.getenv("MANAGER_TIMEOUT_IN_HOURS", 12))  # Default to 12 hours if not set

# Ensure both parameters are present
if not MANAGER_START_DATE or not MANAGER_REGISTRY_FOLDER:
    raise ValueError("Both MANAGER_START_DATE and MANAGER_REGISTRY_FOLDER must be provided in the .env file.")

# Convert MANAGER_START_DATE to datetime
try:
    MANAGER_START_DATE = datetime.strptime(MANAGER_START_DATE, "%Y-%m-%d")
except ValueError:
    raise ValueError("MANAGER_START_DATE must be in the format YYYY-MM-DD.")

# Expand user paths
REGISTRY_FOLDER = os.path.expanduser(os.getenv("MANAGER_REGISTRY_FOLDER"))
if not REGISTRY_FOLDER:
    raise ValueError("Environment variable MANAGER_REGISTRY_FOLDER must be defined in the .env file.")

# Ensure the directory exists
os.makedirs(REGISTRY_FOLDER, exist_ok=True)

# Define the path to the registry file
REGISTRY_FILE = os.path.join(REGISTRY_FOLDER, 'manager_registry.json')

# Define the temporal scope end date (e.g., today)
END_DATE = datetime.today()

def initialize_registry():
    """Initialize a new registry JSON file if it doesn't exist."""
    if not os.path.exists(REGISTRY_FILE):
        # Create the registry file with an empty list for processed Fridays
        with open(REGISTRY_FILE, 'w') as f:
            json.dump({"processed_fridays": []}, f)
    else:
        print("Registry file already exists.")

def get_processed_fridays():
    """Read the processed Fridays from the registry JSON file."""
    with open(REGISTRY_FILE, 'r') as f:
        registry_data = json.load(f)
    return registry_data.get("processed_fridays", [])

def save_processed_friday(friday):
    """Add a Friday to the registry as processed."""
    processed_fridays = get_processed_fridays()
    if friday not in processed_fridays:
        processed_fridays.append(friday)
        with open(REGISTRY_FILE, 'w') as f:
            json.dump({"processed_fridays": processed_fridays}, f)

def get_last_processed_friday():
    """Get the last processed Friday from the registry."""
    processed_fridays = get_processed_fridays()
    if processed_fridays:
        # Convert the last processed Friday string to a datetime object
        last_friday_str = max(processed_fridays)
        last_friday = datetime.strptime(last_friday_str, "%Y-%m-%d")
        return last_friday
    else:
        # Return the start date if nothing is processed
        return MANAGER_START_DATE


def find_next_friday(last_processed_friday):
    """Find the next Friday after the last processed one."""
    next_friday = last_processed_friday + timedelta(days=(4 - last_processed_friday.weekday() + 7) % 7)
    return next_friday

def trigger_scraping_dag_for_next_friday(**kwargs):
    """Trigger the scraping DAG for the first Friday that hasn't been processed."""
    # Get the last processed Friday from the registry
    last_processed_friday = get_last_processed_friday()
    
    # Find the next Friday to process
    next_friday = find_next_friday(last_processed_friday)
    
    # Check if the next Friday is after the current system date (END_DATE)
    if next_friday > END_DATE:
        # Log a message and skip processing if it's after the current date
        logging.debug(f"Next Friday ({next_friday}) is after the current system date ({END_DATE}). No scraping needed.")
        return None  # No action needed
    
    # Check if this Friday is already processed, if not trigger scraping DAG
    processed_fridays = get_processed_fridays()
    
    if next_friday.strftime('%Y-%m-%d') not in processed_fridays:
        logging.debug(f"Triggering scraping for Friday: {next_friday}")
        kwargs['dag_run'].conf = {'execution_date': next_friday}
        save_processed_friday(next_friday.strftime('%Y-%m-%d'))  # Mark this Friday as processed
        return next_friday
    else:
        logging.debug(f"Friday {next_friday} has already been processed. Skipping scraping.")
        return None  # No new Friday to scrape

default_args = {
    'owner': 'airflow',
    'start_date': MANAGER_START_DATE,
    'retries': 2,  # Cap the number of retry attempts
    'retry_delay': timedelta(seconds=15),  # Optional: delay between retries
}

# Define the Manager DAG
with DAG(
    'manager_dag',
    default_args=default_args,
    schedule_interval='@daily',
) as dag:

    # Task to initialize the registry if it doesn't exist
    initialize_registry_task = PythonOperator(
        task_id='initialize_registry',
        python_callable=initialize_registry
    )
    
    check_and_trigger_scraping = PythonOperator(
        task_id='check_and_trigger_scraping',
        python_callable=trigger_scraping_dag_for_next_friday,
        provide_context=True,
    )
    
    # External Task Sensor to wait for the worker DAG's completion within the timeout period
    wait_for_scraping_completion = ExternalTaskSensor(
        task_id='wait_for_scraping_completion',
        external_dag_id='job_listing_scraper_1.2',
        timeout=MANAGER_TIMEOUT_IN_HOURS * 3600,  # Timeout in seconds
        poke_interval=60,  # Check every 60 seconds
        mode='poke',  # Can be 'poke' or 'reschedule'
    )
    
    trigger_scraping_dag = TriggerDagRunOperator(
        task_id='trigger_scraping_dag',
        trigger_dag_id='job_listing_scraper_1.2',
        conf={
            'execution_date': '{{ task_instance.xcom_pull(task_ids="check_and_trigger_scraping") }}'
        },
    )

    # Define the task sequence
    initialize_registry_task >> check_and_trigger_scraping >> trigger_scraping_dag >> wait_for_scraping_completion
