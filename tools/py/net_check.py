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