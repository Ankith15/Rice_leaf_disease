import os
from dotenv import load_dotenv

load_dotenv()
run = os.getenv("RUN")
print(run)