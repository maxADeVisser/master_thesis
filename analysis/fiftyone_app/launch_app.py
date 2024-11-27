import fiftyone as fo
from dotenv import load_dotenv

load_dotenv()

session = fo.launch_app(port=5151)
session.wait()
