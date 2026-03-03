import sys
import json
import urllib.request
import urllib.error
import os

SERVER_URL = "http://127.0.0.1:8765/rpc"

def log(msg):
    # Log to stderr so it doesn't interfere with MCP stdio
    sys.stderr.write(f"[PROXY] {msg}\n")
    sys.stderr.flush()

def main():
    log("Started MCP Proxy")
    
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
                
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log(f"Invalid JSON: {line[:50]}...")
                continue
                
            # Forward to server
            try:
                req = urllib.request.Request(
                    SERVER_URL,
                    data=json.dumps(message).encode('utf-8'),
                    headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req) as resp:
                    if resp.status == 204:
                         # Notification or no result
                         continue
                         
                    response_data = resp.read()
                    
                    # Write response to stdout for MCP client
                    sys.stdout.buffer.write(response_data)
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    
            except urllib.error.URLError as e:
                log(f"Connection error: {e}. Is mcp_server_http.py running?")
                # Optionally send an error back to client if it was a request
                if "id" in message:
                    err_response = {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32000, "message": f"Proxy connection failed: {e}"}
                    }
                    sys.stdout.write(json.dumps(err_response) + "\n")
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
