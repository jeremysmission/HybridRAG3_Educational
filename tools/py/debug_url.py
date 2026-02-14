import sys
import os
sys.path.insert(0, os.getcwd())

try:
    import keyring
    endpoint = keyring.get_password("hybridrag", "azure_endpoint")
    api_key = keyring.get_password("hybridrag", "azure_api_key")
except ImportError:
    print("  [ERROR] keyring not installed. Run: pip install keyring")
    sys.exit(1)

if not endpoint:
    print("  [ERROR] No endpoint stored. Run: rag-store-endpoint")
    sys.exit(1)
if not api_key:
    print("  [ERROR] No API key stored. Run: rag-store-key")
    sys.exit(1)

print(f"  Stored endpoint:   {endpoint}")
print(f"  Key preview:       {api_key[:4]}...{api_key[-4:]}")
print()

# Detect provider (recognizes aoai, azure, and other patterns)
url_lower = endpoint.lower()
is_azure = (
    "azure" in url_lower
    or ".openai.azure.com" in url_lower
    or "aoai" in url_lower
    or "azure-api" in url_lower
    or "cognitiveservices" in url_lower
)

provider = "AZURE" if is_azure else "OpenAI"
markers = [m for m in ["azure", ".openai.azure.com", "aoai", "azure-api", "cognitiveservices"] 
           if m in url_lower]
print(f"  Provider detected: {provider}")
print(f"  Matched on:        {', '.join(markers) if markers else 'NONE (defaulting to OpenAI)'}")
print()

# Check env vars for deployment and version
deployment = None
api_version = None
for var in ["AZURE_OPENAI_DEPLOYMENT", "AZURE_DEPLOYMENT", "OPENAI_DEPLOYMENT",
            "AZURE_OPENAI_DEPLOYMENT_NAME", "DEPLOYMENT_NAME", "AZURE_CHAT_DEPLOYMENT"]:
    val = os.environ.get(var)
    if val:
        deployment = val
        print(f"  Deployment:        {deployment} (from env: {var})")
        break

for var in ["AZURE_OPENAI_API_VERSION", "AZURE_API_VERSION", "API_VERSION"]:
    val = os.environ.get(var)
    if val:
        api_version = val
        print(f"  API version:       {api_version} (from env: {var})")
        break

# Extract deployment from URL if present
import re
if "/deployments/" in endpoint and not deployment:
    match = re.search(r"/deployments/([^/]+)", endpoint)
    if match:
        deployment = match.group(1)
        print(f"  Deployment:        {deployment} (from URL)")

# Build URL
base = endpoint.rstrip("/")
if is_azure:
    if "/chat/completions" in endpoint:
        final_url = base
        if "api-version" not in base:
            v = api_version or "2024-02-01"
            final_url = f"{base}?api-version={v}"
        strategy = "Complete URL -- using as-is"
    elif "/deployments/" in endpoint:
        v = api_version or "2024-02-01"
        final_url = f"{base}/chat/completions?api-version={v}"
        strategy = "Has deployment -- appending /chat/completions"
    else:
        d = deployment or "gpt-35-turbo"
        v = api_version or "2024-02-01"
        if not deployment:
            print(f"  Deployment:        {d} (GUESSED -- run rag-store-deployment to set)")
        if not api_version:
            print(f"  API version:       {v} (default -- run rag-store-api-version to set)")
        final_url = f"{base}/openai/deployments/{d}/chat/completions?api-version={v}"
        strategy = "Built full Azure path from base URL"
    auth_header = "api-key"
else:
    if "/chat/completions" in endpoint:
        final_url = base
    else:
        final_url = f"{base}/v1/chat/completions"
    auth_header = "Authorization: Bearer"
    strategy = "Standard OpenAI path"

print()
print(f"  Strategy:          {strategy}")
print(f"  Auth header:       {auth_header}")
print(f"  Final URL:         {final_url}")
print()

# Check problems
problems = []
clean = final_url.replace("https://", "").replace("http://", "")
if "//" in clean:
    problems.append("DOUBLE SLASH in URL path")
if is_azure and "v1/chat" in final_url:
    problems.append("OpenAI path on Azure endpoint")
if is_azure and not deployment and "/deployments/" not in endpoint and "/chat/completions" not in endpoint:
    problems.append("No deployment name -- guessing gpt-35-turbo (may be wrong)")

if problems:
    print("  PROBLEMS:")
    for p in problems:
        print(f"    [!] {p}")
else:
    print("  No problems detected.")
print()
print("  Next: run rag-test-api-verbose to make a live test call.")