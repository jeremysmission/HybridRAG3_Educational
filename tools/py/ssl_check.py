import ssl, socket, keyring
from urllib.parse import urlparse

endpoint = keyring.get_password("hybridrag", "azure_endpoint")
if not endpoint:
    print("  No endpoint stored. Run rag-store-endpoint first.")
    exit()

parsed = urlparse(endpoint)
host = parsed.hostname

print(f"  Testing SSL to: {host}")
print(f"  Python SSL version: {ssl.OPENSSL_VERSION}")
print()

ctx = ssl.create_default_context()
try:
    with ctx.wrap_socket(socket.socket(), server_hostname=host) as s:
        s.settimeout(10)
        s.connect((host, 443))
        cert = s.getpeercert()
        print(f"  [OK] SSL handshake successful")
        print(f"  Protocol: {s.version()}")
        print(f"  Cipher:   {s.cipher()[0]}")
        if cert:
            subject = dict(x[0] for x in cert.get("subject", []))
            issuer = dict(x[0] for x in cert.get("issuer", []))
            print(f"  Subject:  {subject.get('commonName', 'unknown')}")
            print(f"  Issuer:   {issuer.get('organizationName', 'unknown')}")
            print(f"  Expires:  {cert.get('notAfter', 'unknown')}")

except ssl.SSLCertVerificationError as e:
    print(f"  [FAIL] Certificate verification failed")
    print(f"  Error: {e}")
    print()
    print("  This usually means a corporate proxy is intercepting HTTPS.")
    print("  Ask IT for the corporate CA certificate, then:")
    print("    $env:REQUESTS_CA_BUNDLE = 'path\\to\\corporate-ca.pem'")

except Exception as e:
    print(f"  [FAIL] {e}")