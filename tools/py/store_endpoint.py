import sys
import keyring

endpoint = sys.argv[1]
keyring.set_password("hybridrag", "azure_endpoint", endpoint)
stored = keyring.get_password("hybridrag", "azure_endpoint")
if stored == endpoint:
    print("  [OK] Endpoint stored successfully.")
    print("  Value: " + stored)
else:
    print("  [ERROR] Endpoint storage failed.")
