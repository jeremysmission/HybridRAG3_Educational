# ============================================================================
# HybridRAG -- Ollama Connection Test (tools/py/ollama_test.py)
# ============================================================================
#
# WHAT THIS DOES:
#   Sends a simple test prompt to your local Ollama server (the offline
#   AI model running on your own machine) and shows the response. This
#   verifies that Ollama is running, the phi4-mini model is loaded, and
#   it can generate text.
#
# WHAT OLLAMA IS:
#   Ollama is a program that runs AI models locally on your computer --
#   no internet needed. Think of it as having a small AI brain running
#   on your own hardware instead of calling a cloud service.
#
# HOW TO USE:
#   1. Start Ollama first: ollama serve (or rag-ollama-start)
#   2. Run this: python tools/py/ollama_test.py
#
# EXPECTED OUTPUT:
#   Latency: 2-15 seconds (depends on your hardware)
#   Response: Some 3-word greeting
#   [SUCCESS] Ollama is working!
#
# IF IT FAILS:
#   "Cannot connect" = Ollama isn't running. Start it first.
# ============================================================================
import json, urllib.request, time

url = "http://localhost:11434/api/generate"
payload = {
    "model": "phi4-mini",
    "prompt": "Say hello in exactly 3 words.",
    "stream": False
}

print("  Sending test to Ollama (phi4-mini)...")
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