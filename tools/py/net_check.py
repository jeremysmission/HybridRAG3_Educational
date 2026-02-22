# ============================================================================
# HybridRAG -- Network Connectivity Check (tools/py/net_check.py)
# ============================================================================
#
# WHAT THIS DOES:
#   Tests whether your machine can reach the services HybridRAG needs.
#   Like pinging a radio repeater to see if you have line-of-sight --
#   this checks if you have a clear network path to each service.
#
# SERVICES TESTED:
#   1. Azure API    -- Your configured AI endpoint (port 443/HTTPS)
#   2. Ollama       -- Local AI model server (port 11434, localhost)
#   3. HuggingFace  -- Model downloads (port 443/HTTPS)
#   4. PyPI         -- Python package installs (port 443/HTTPS)
#   5. GitHub       -- Code repository (port 443/HTTPS)
#
# OUTPUT:
#   [OK]   = connection succeeded, shows latency in milliseconds
#   [FAIL] = cannot reach the service (DNS failed or timeout)
#   [DOWN] = service refused the connection (it's there but not listening)
#
# HOW TO USE:
#   python tools/py/net_check.py
#
# COMMON RESULTS:
#   - All [OK] except Ollama: normal if you haven't started Ollama yet
#   - All [FAIL]: you're probably on a restricted network or VPN is down
#   - Azure [FAIL], others [OK]: endpoint URL may be wrong
# ============================================================================
import keyring, socket, ssl, time

endpoint = keyring.get_password("hybridrag", "azure_endpoint")

# Parse hostname from endpoint
targets = []
if endpoint:
    from urllib.parse import urlparse
    parsed = urlparse(endpoint)
    if parsed.hostname:
        targets.append(("Azure API", parsed.hostname, 443))

targets += [
    ("Ollama (local)", "127.0.0.1", 11434),
    ("HuggingFace", "huggingface.co", 443),
    ("PyPI", "pypi.org", 443),
    ("GitHub", "github.com", 443),
]

for name, host, port in targets:
    try:
        start = time.time()
        sock = socket.create_connection((host, port), timeout=5)
        latency = (time.time() - start) * 1000
        sock.close()
        print(f"  [OK]   {name:20s} {host}:{port}  ({latency:.0f}ms)")
    except socket.timeout:
        print(f"  [FAIL] {name:20s} {host}:{port}  (timeout)")
    except ConnectionRefusedError:
        print(f"  [DOWN] {name:20s} {host}:{port}  (refused)")
    except socket.gaierror:
        print(f"  [FAIL] {name:20s} {host}:{port}  (DNS failed)")
    except Exception as e:
        print(f"  [FAIL] {name:20s} {host}:{port}  ({e})")