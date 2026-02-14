import sys
import keyring

key = sys.argv[1]
keyring.set_password("hybridrag", "azure_api_key", key)
stored = keyring.get_password("hybridrag", "azure_api_key")
if stored == key:
    print("  [OK] API key stored successfully.")
    print("  Preview: " + stored[:4] + "..." + stored[-4:])
else:
    print("  [ERROR] Key storage failed. Keyring may not be available.")
