import json
import urllib.request

# Test message
message = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "rag_search",
        "arguments": {
            "query_text": "stiamo lavorando su bpofh",
            "top_k": 3
        }
    }
}

url = "http://127.0.0.1:8765/rpc"

print(f"Testing {url}...")
print(f"Request: {json.dumps(message, indent=2)}\n")

try:
    req = urllib.request.Request(
        url,
        data=json.dumps(message).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req) as resp:
        response_data = resp.read().decode('utf-8')
        response = json.loads(response_data)
        
        print(f"Status: {resp.status}")
        print(f"Response:\n{json.dumps(response, indent=2)}")
        
        if "result" in response:
            print("\n✓ SUCCESS: Got result data!")
            if "content" in response["result"]:
                content = response["result"]["content"][0]["text"]
                data = json.loads(content)
                print(f"  - Found {len(data.get('results', []))} results")
        else:
            print("\n✗ ERROR: No result in response")
            
except urllib.error.URLError as e:
    print(f"✗ Connection failed: {e}")
    print("  Make sure start_mcp_server.bat is running!")
except Exception as e:
    print(f"✗ Error: {e}")
