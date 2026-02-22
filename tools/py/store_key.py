# ============================================================================
# HybridRAG -- Store Azure API Key (tools/py/store_key.py)
# ============================================================================
#
# WHAT THIS DOES:
#   Saves your Azure OpenAI API key into Windows Credential Manager so
#   HybridRAG can use it later without you having to type it every time.
#   Think of it like saving a Wi-Fi password -- you enter it once, and
#   Windows remembers it securely.
#
# HOW TO USE:
#   python tools/py/store_key.py YOUR_API_KEY_HERE
#
# WHERE THE KEY GOES:
#   Windows Credential Manager -> Generic Credentials -> "hybridrag"
#   You can view it in Control Panel -> Credential Manager if needed.
#
# SAFETY:
#   - The key is stored encrypted by Windows, not in a plain text file
#   - It never appears in git, logs, or config files
#   - Only your Windows user account can read it back
# ============================================================================
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
