import keyring

items = {
    "azure_endpoint": "Endpoint",
    "azure_api_key": "API Key",
}

for key_name, display_name in items.items():
    val = keyring.get_password("hybridrag", key_name)
    if val:
        if "key" in key_name.lower():
            display = val[:4] + "..." + val[-4:] + f"  (length: {len(val)})"
        else:
            display = val
        print(f"  {display_name}: {display}")
    else:
        print(f"  {display_name}: NOT SET")