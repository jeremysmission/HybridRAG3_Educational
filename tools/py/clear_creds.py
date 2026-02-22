# ============================================================================
# clear_creds.py -- Remove all stored credentials from keyring
# ============================================================================
# Clears all four credential values from Windows Credential Manager:
#   - azure_api_key
#   - azure_endpoint
#   - azure_deployment
#   - azure_api_version
#
# INTERNET ACCESS: NONE -- only modifies local keyring
# ============================================================================

import keyring

for key_name in ["azure_api_key", "azure_endpoint",
                 "azure_deployment", "azure_api_version"]:
    try:
        keyring.delete_password("hybridrag", key_name)
        print(f"  [OK] Deleted: {key_name}")
    except keyring.errors.PasswordDeleteError:
        print(f"  [--] Not found: {key_name} (already deleted or never set)")
    except Exception as e:
        print(f"  [FAIL] {key_name}: {e}")
