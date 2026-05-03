import urllib.request
import ssl
from dotenv import load_dotenv
import os

load_dotenv()
url = os.getenv("QDRANT_URL", "").rstrip("/")
api_key = os.getenv("QDRANT_API_KEY", "")

# Try hitting root endpoint with API key
for endpoint in ["/", "/collections", "/healthz"]:
    full_url = url + endpoint
    req = urllib.request.Request(full_url)
    req.add_header("api-key", api_key)
    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            print(f"GET {endpoint} -> {r.status}: {r.read(200).decode()}")
    except urllib.error.HTTPError as e:
        print(f"GET {endpoint} -> HTTP {e.code}: {e.read(200).decode()}")
    except Exception as e:
        print(f"GET {endpoint} -> ERROR: {e}")
