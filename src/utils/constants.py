import torch
from dotenv import load_dotenv
import os

# Automatically finds and loads the .env file
load_dotenv()

DEFAULT_DATA_DIR = "data/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read the STDOUT environment variable
STDOUT = os.environ.get("STDOUT", "False").lower() in ("true", "1", "t")

# Crop yield statistics (mean and std for normalization)
CROP_YIELD_STATS = {
    "soybean": {
        "mean": [],
        "std": [],
    },
    "corn": {
        "mean": [],
        "std": [],
    },
}

TOTAL_WEATHER_VARS = 31
MAX_GRANULARITY_DAYS = 31
MAX_CONTEXT_LENGTH = 364
NUM_DATASET_PARTS = 119
VALIDATION_CHUNK_IDS = [7, 30, 56, 59, 93, 106, 110, 24]

# Test years for cross-validation
TEST_YEARS = [2014, 2015, 2016, 2017, 2018]

EXTREME_YEARS = {
    "usa": {
        "corn": [2002, 2004, 2009, 2012, 2014],
        "soybean": [2003, 2004, 2009, 2012, 2016],
    },
}
