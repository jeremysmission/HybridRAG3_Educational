import sys
print(f"  Python:             {sys.version.split()[0]}")

packages = [
    "sentence_transformers", "torch", "numpy", "keyring",
    "structlog", "yaml", "requests", "urllib3", "certifi",
    "sqlite3"
]

for pkg in packages:
    try:
        if pkg == "yaml":
            import yaml
            print(f"  PyYAML:             {yaml.__version__}")
        elif pkg == "sqlite3":
            import sqlite3
            print(f"  SQLite:             {sqlite3.sqlite_version}")
        else:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed (no version)")
            display = pkg.replace("_", "-")
            print(f"  {display:20s} {ver}")
    except ImportError:
        display = pkg.replace("_", "-")
        print(f"  {display:20s} NOT INSTALLED")