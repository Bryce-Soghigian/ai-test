import schedule
import time
from datetime import datetime
import os
import subprocess
import logging
from logging.handlers import RotatingFileHandler

# Set up logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    handlers=[
        RotatingFileHandler(
            "logs/scheduler.log",
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
        ),
        logging.StreamHandler(),
    ],
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("recommendation_scheduler")

def run_retraining():
    """Run the model retraining script."""
    try:
        logger.info("Starting model retraining...")
        start_time = time.time()
        
        # Run the retraining script
        result = subprocess.run(
            ["python", "-m", "src.retrain"],
            capture_output=True,
            text=True,
            check=True,
        )
        
        duration = time.time() - start_time
        logger.info(f"Model retraining completed in {duration:.2f} seconds")
        logger.debug(f"Retraining output:\n{result.stdout}")
        
        # Check if models were updated
        if "Saving improved models" in result.stdout:
            logger.info("Models were improved and updated")
        else:
            logger.info("No model improvements found")
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed with error:\n{e.stderr}")
    except Exception as e:
        logger.error(f"Unexpected error during retraining: {str(e)}")

def run_metrics_backup():
    """Backup metrics data to persistent storage."""
    try:
        logger.info("Starting metrics backup...")
        
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"metrics/backup_{timestamp}"
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy current metrics file to backup
        if os.path.exists("metrics/recommendation_metrics.csv"):
            subprocess.run([
                "cp",
                "metrics/recommendation_metrics.csv",
                f"{backup_dir}/recommendation_metrics.csv",
            ], check=True)
            
            logger.info(f"Metrics backed up to {backup_dir}")
        else:
            logger.warning("No metrics file found to backup")
    
    except Exception as e:
        logger.error(f"Metrics backup failed: {str(e)}")

def main():
    # Schedule model retraining
    schedule.every().day.at("02:00").do(run_retraining)  # Run at 2 AM daily
    
    # Schedule metrics backup
    schedule.every().hour.do(run_metrics_backup)  # Backup metrics every hour
    
    logger.info("Scheduler started")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Scheduler error: {str(e)}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    main() 