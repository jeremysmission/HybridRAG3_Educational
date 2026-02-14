import keyring

for key_name in ["azure_api_key", "azure_endpoint"]:
    try:
        keyring.delete_password("hybridrag", key_name)
        print(f"  [OK] Deleted: {key_name}")
    except keyring.errors.PasswordDeleteError:
        print(f"  [--] Not found: {key_name} (already deleted or never set)")
    except Exception as e:
        print(f"  [ERROR] {key_name}: {e}")