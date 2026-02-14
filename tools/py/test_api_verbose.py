import sys, os, json, time
sys.path.insert(0, os.getcwd())

import keyring
endpoint = keyring.get_password("hybridrag", "azure_endpoint")
api_key = keyring.get_password("hybridrag", "azure_api_key")

if not endpoint or not api_key:
    print("  [ERROR] Missing credentials. Run rag-store-endpoint and rag-store-key.")
    sys.exit(1)

url_lower = endpoint.lower()
is_azure = ("azure" in url_lower or ".openai.azure.com" in url_lower 
            or "aoai" in url_lower or "azure-api" in url_lower
            or "cognitiveservices" in url_lower)

base = endpoint.rstrip("/")

if is_azure:
    if "/chat/completions" in endpoint:
        final_url = base
        if "api-version" not in base:
            final_url += "?api-version=" + os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    elif "/deployments/" in endpoint:
        v = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        final_url = f"{base}/chat/completions?api-version={v}"
    else:
        d = None
        for var in ["AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT", "OPENAI_DEPLOYMENT",
                     "AZURE_OPENAI_DEPLOYMENT_NAME", "DEPLOYMENT_NAME"]:
            val = os.environ.get(var)
            if val:
                d = val
                break
        if not d:
            d = "gpt-35-turbo"
        v = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        final_url = f"{base}/openai/deployments/{d}/chat/completions?api-version={v}"
    
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    print(f"  Provider:  AZURE")
    print(f"  Auth:      api-key header")
else:
    if "/chat/completions" in endpoint:
        final_url = base
    else:
        final_url = f"{base}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    print(f"  Provider:  OpenAI")
    print(f"  Auth:      Bearer token")

print(f"  URL:       {final_url}")
print()

payload = {
    "messages": [{"role": "user", "content": "Say hello in exactly 3 words."}],
    "max_tokens": 20,
    "temperature": 0.1
}

print("  Sending request...")
import urllib.request, urllib.error, ssl

req = urllib.request.Request(final_url, data=json.dumps(payload).encode("utf-8"),
                             headers=headers, method="POST")
ctx = ssl.create_default_context()
start = time.time()

try:
    response = urllib.request.urlopen(req, context=ctx, timeout=30)
    latency = time.time() - start
    body = json.loads(response.read().decode("utf-8"))
    
    print(f"  Status:    200 OK")
    print(f"  Latency:   {latency:.2f}s")
    if "choices" in body and body["choices"]:
        print(f"  Response:  {body['choices'][0]['message']['content']}")
        print(f"  Model:     {body.get('model', 'unknown')}")
    if "usage" in body:
        u = body["usage"]
        print(f"  Tokens:    {u.get('prompt_tokens','?')} in, {u.get('completion_tokens','?')} out")
    print()
    print("  [SUCCESS] API is working!")

except urllib.error.HTTPError as e:
    latency = time.time() - start
    error_body = ""
    try: error_body = e.read().decode("utf-8")
    except: pass
    print(f"  Status:    {e.code} {e.reason}")
    print(f"  Latency:   {latency:.2f}s")
    if error_body: print(f"  Response:  {error_body[:500]}")
    print()
    if e.code == 401:
        print("  [FAIL] Auth error. Key may be wrong/expired, or header format is wrong.")
    elif e.code == 404:
        print("  [FAIL] Not found. Deployment name or API version may be wrong.")
        print("  >> Run rag-store-deployment to set the correct name.")
    elif e.code == 403:
        print("  [FAIL] Forbidden. Key may lack permission for this deployment.")
    elif e.code == 429:
        print("  [FAIL] Rate limited. Wait a minute.")
    else:
        print(f"  [FAIL] HTTP {e.code}")

except urllib.error.URLError as e:
    latency = time.time() - start
    print(f"  [FAIL] Connection error: {e.reason}")
    print("  >> Check VPN/proxy/network settings.")

except Exception as e:
    print(f"  [FAIL] Unexpected: {e}")