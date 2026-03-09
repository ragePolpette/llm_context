
import sys
import os
import json
import asyncio
import uuid
from typing import Any

# Force project path for local imports
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Mock logic from mcp_server_http.py
class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

# Create a dummy payload with UUID to test serialization
payload = {
    "context": "test context",
    "results": [
        {"id": uuid.uuid4(), "content": "test result"}
    ],
    "meta": {"id": uuid.uuid4()}
}

print("Testing UUID serialization...")
try:
    json_str = json.dumps(payload, indent=2, ensure_ascii=True, cls=UUIDEncoder)
    print("Serialization successful!")
    print(json_str)
except Exception as e:
    print(f"Serialization failed: {e}")

# Now test the real module import and logic (dry run)
print("\nTesting mcp_server_http import...")
try:
    import mcp_server_http
    print("Import successful.")
    
    # We can't easily run the full handler without env vars, but we verified the class is there.
    if hasattr(mcp_server_http, 'UUIDEncoder'):
        print("UUIDEncoder class found in module.")
    else:
        print("ERROR: UUIDEncoder class NOT found in module.")
        
except Exception as e:
    print(f"Import failed: {e}")
