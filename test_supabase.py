
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

client = create_client(url, key)

try:
    response = client.storage.list_buckets()
    print("SUCCESS: Supabase API is reachable.")
    print("Storage API responded successfully.")
except Exception as e:
    print("API request failed:", type(e).__name__, str(e))
