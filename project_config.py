"""Project config file"""

import os

from dotenv import load_dotenv

load_dotenv(".env")


class _Config:
    """Config class for the project. Intended to be a singleton"""

    def __init__(self):
        self.DATA_DIR = os.getenv("LIDC_IDRI_DIR")
        self.patient_ids = sorted(
            [
                pid
                for pid in os.listdir(self.DATA_DIR)
                if os.path.isdir(os.path.join(self.DATA_DIR, pid))
            ]
        )


# Singleton pattern: only one instance of the Config class is created
config = _Config()

if __name__ == "__main__":
    config = _Config()
