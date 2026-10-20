"""Test connectivity to Supabase storage."""

import os

try:
    from dotenv import load_dotenv
    from supabase import create_client
except ImportError:
    load_dotenv = None
    create_client = None


def test_supabase_connection():
    if not load_dotenv or not create_client:
        print("Note: 'supabase' and 'python-dotenv' are required to test Supabase.")
        print("Install them with: pip install supabase python-dotenv")
        return False

    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        print("SUPABASE_URL or SUPABASE_KEY is not set in environment.")
        return False

    client = create_client(url, key)

    try:
        response = client.storage.list_buckets()
        print("SUCCESS: Supabase API is reachable.")
        print("Storage API responded successfully.")
        return True
    except Exception as e:
        print("API request failed:", type(e).__name__, str(e))
        return False


if __name__ == "__main__":
    test_supabase_connection()
