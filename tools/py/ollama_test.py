import json, urllib.request, time

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3",
    "prompt": "Say hello in exactly 3 words.",
    "stream": False
}

print("  Sending test to Ollama (llama3)...")
start = time.time()

try:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    response = urllib.request.urlopen(req, timeout=120)
    latency = time.time() - start
    body = json.loads(response.read().decode("utf-8"))
    
    print(f"  Latency:   {latency:.1f}s")
    print(f"  Response:  {body.get('response', 'no response field')}")
    print(f"  Model:     {body.get('model', 'unknown')}")
    print()
    print("  [SUCCESS] Ollama is working!")

except urllib.error.URLError as e:
    print(f"  [FAIL] Cannot connect to Ollama: {e.reason}")
    print("  >> Is Ollama running? Run: rag-ollama-start")
except Exception as e:
    print(f"  [FAIL] {e}")