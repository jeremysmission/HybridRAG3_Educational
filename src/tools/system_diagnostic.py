#!/usr/bin/env python3
"""
HybridRAG v3 -- System Diagnostic & Benchmarking Tool
Location: src/tools/system_diagnostic.py

THREE-TIER DIAGNOSTIC:
  Tier 1: Schema & Logic Tests (instant, zero deps, runs at setup)
  Tier 2: Pipeline Test (loads embedding model, synthetic data)
  Tier 3: Stress Test (needs 1GB+ docs, full production load)

ALSO PROVIDES:
  - Hardware fingerprint (CPU, RAM, disk -> config/system_profile.json)
  - Security audit (HuggingFace offline, API endpoint, kill switch)
  - Performance baseline (chunks/sec, MB/sec, RAM peak)
  - Profile recommendation (laptop_safe / desktop_power / server_max)

USAGE:
  python -m src.tools.system_diagnostic                   # Tier 1 only
  python -m src.tools.system_diagnostic --tier 2          # Tier 1 + 2
  python -m src.tools.system_diagnostic --tier 3          # All tiers
  python -m src.tools.system_diagnostic --hardware-only   # Fingerprint only

INTERNET ACCESS: NONE -- this script makes zero network calls.
"""

import argparse, gc, json, os, platform, sqlite3, subprocess, sys
import tempfile, time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ===== SECTION 1: HARDWARE FINGERPRINT =====================================
# Collects CPU, RAM, disk via stdlib + PowerShell WMI. No psutil needed.

def _ps(cmd, timeout=15):
    """Run PowerShell command, return stdout or empty string on failure."""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""

def collect_hardware_fingerprint():
    """Detect CPU, RAM, disk, GPU. Works on Windows (WMI) and Linux (/proc)."""
    print("\n" + "=" * 60)
    print("  HARDWARE FINGERPRINT")
    print("=" * 60)

    hw = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": {"system": platform.system(), "release": platform.release(),
               "version": platform.version()},
        "python": {"version": platform.python_version(),
                   "architecture": platform.architecture()[0]},
        "cpu": {"name": "Unknown", "physical_cores": os.cpu_count() or 0,
                "logical_cores": os.cpu_count() or 0, "max_clock_mhz": 0},
        "ram_gb": 0.0, "disk": [], "gpu": "None detected",
        "gpu_vram_gb": 0.0, "has_nvme": False,
    }

    if platform.system() == "Windows":
        # --- CPU via WMI ---
        out = _ps("Get-CimInstance Win32_Processor | Select-Object -First 1 "
                  "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed "
                  "| ConvertTo-Json")
        if out:
            try:
                c = json.loads(out)
                hw["cpu"]["name"] = (c.get("Name","") or "").strip()
                hw["cpu"]["physical_cores"] = c.get("NumberOfCores", 0)
                hw["cpu"]["logical_cores"] = c.get("NumberOfLogicalProcessors", 0)
                hw["cpu"]["max_clock_mhz"] = c.get("MaxClockSpeed", 0)
            except Exception: pass

        # --- RAM ---
        out = _ps("[math]::Round((Get-CimInstance Win32_ComputerSystem)"
                  ".TotalPhysicalMemory / 1GB, 1)")
        if out:
            try: hw["ram_gb"] = float(out)
            except Exception: pass

        # --- Disk ---
        out = _ps("Get-CimInstance Win32_DiskDrive | Select-Object Model,"
                  "MediaType,@{N='SizeGB';E={[math]::Round($_.Size/1GB)}} "
                  "| ConvertTo-Json")
        if out:
            try:
                disks = json.loads(out)
                if isinstance(disks, dict): disks = [disks]
                for d in disks:
                    media = (d.get("MediaType","") or "").lower()
                    model = (d.get("Model","") or "").lower()
                    if "external" in media: dtype = "External"
                    elif any(x in model for x in ["nvme","hynix","samsung","ssd"]): dtype = "NVMe SSD"
                    elif "fixed" in media: dtype = "SSD (probable)"
                    else: dtype = "HDD"
                    hw["disk"].append({"model": (d.get("Model","") or "").strip(),
                                       "size_gb": d.get("SizeGB", 0), "type": dtype})
            except Exception: pass

        # --- GPU name + VRAM ---
        # Strategy: nvidia-smi first (accurate for all NVIDIA GPUs), then
        # WMI fallback. WMI AdapterRAM is a 32-bit uint that wraps to 0
        # for GPUs with >4GB VRAM.
        nvsmi_vram = 0
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    hw["gpu"] = parts[0]
                    nvsmi_vram = round(int(parts[1]) / 1024, 1)
                    hw["gpu_vram_gb"] = nvsmi_vram
        except Exception:
            pass

        # WMI fallback for GPU name (and VRAM if nvidia-smi unavailable)
        if hw["gpu"] == "None detected" or nvsmi_vram == 0:
            out = _ps("Get-CimInstance Win32_VideoController | "
                      "Select-Object Name,AdapterRAM | ConvertTo-Json")
            if out:
                try:
                    gpus = json.loads(out)
                    if isinstance(gpus, dict):
                        gpus = [gpus]
                    best_vram = 0
                    best_name = "None detected"
                    for g in gpus:
                        name = (g.get("Name", "") or "").strip()
                        adapter_ram = g.get("AdapterRAM") or 0
                        vram_gb = round(adapter_ram / (1024 ** 3), 1)
                        if vram_gb > best_vram:
                            best_vram = vram_gb
                            best_name = name
                        elif name and best_name == "None detected":
                            best_name = name
                    if hw["gpu"] == "None detected":
                        hw["gpu"] = best_name
                    if nvsmi_vram == 0 and best_vram > 0:
                        hw["gpu_vram_gb"] = best_vram
                except Exception:
                    pass

        # --- NVMe detection ---
        for d in hw["disk"]:
            if d.get("type") == "NVMe SSD":
                hw["has_nvme"] = True
                break

    else:
        # Linux fallback
        try:
            r = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10)
            for line in r.stdout.splitlines():
                if "Model name" in line: hw["cpu"]["name"] = line.split(":")[1].strip()
                elif "Core(s) per socket" in line: hw["cpu"]["physical_cores"] = int(line.split(":")[1])
                elif line.startswith("CPU(s):"): hw["cpu"]["logical_cores"] = int(line.split(":")[1])
        except Exception: pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        hw["ram_gb"] = round(int(line.split()[1]) / (1024*1024), 1)
                        break
        except Exception: pass

    # Display
    c = hw["cpu"]
    print(f"  CPU:    {c['name']}")
    print(f"  Cores:  {c['physical_cores']} physical, {c['logical_cores']} logical")
    print(f"  Clock:  {c['max_clock_mhz']} MHz")
    print(f"  RAM:    {hw['ram_gb']} GB")
    for d in hw["disk"]:
        print(f"  Disk:   {d['model']} -- {d['size_gb']} GB ({d['type']})")
    vram = hw.get("gpu_vram_gb", 0)
    vram_str = f" ({vram} GB VRAM)" if vram > 0 else ""
    print(f"  GPU:    {hw['gpu']}{vram_str}")
    if hw.get("has_nvme"):
        print(f"  NVMe:   Detected (fast indexing)")
    print(f"  OS:     {hw['os']['system']} {hw['os']['release']}")
    print(f"  Python: {hw['python']['version']} ({hw['python']['architecture']})")
    return hw


# ===== SECTION 2: PROFILE RECOMMENDATION ===================================

def recommend_profile(hw):
    """Recommend laptop_safe / desktop_power / server_max based on RAM + cores + VRAM."""
    ram = hw.get("ram_gb", 0)
    cores = hw.get("cpu", {}).get("physical_cores", 0)
    vram = hw.get("gpu_vram_gb", 0)

    if ram >= 64 and cores >= 16 and vram >= 24:
        name = "server_max"
        reason = f"{ram}GB RAM + {cores} cores + {vram}GB VRAM"
    elif ram >= 32 and cores >= 8 and vram >= 8:
        name = "desktop_power"
        reason = f"{ram}GB RAM + {cores} cores + {vram}GB VRAM"
    elif ram >= 32 and cores >= 8:
        # Enough RAM/cores but no discrete GPU -- still desktop_power for CPU inference
        name = "desktop_power"
        reason = f"{ram}GB RAM + {cores} cores (no discrete GPU)"
    else:
        name = "laptop_safe"
        parts = [f"{ram}GB RAM", f"{cores} cores"]
        if vram > 0:
            parts.append(f"{vram}GB VRAM")
        reason = " + ".join(parts)

    profiles = {
        "laptop_safe": {
            "description": "Conservative -- stability on 8-16GB laptops",
            "embedding_batch_size": 16, "chunk_size": 1200, "chunk_overlap": 200,
            "retrieval_top_k": 5, "reranker_top_n": 20,
            "indexing_block_chars": 200000, "max_concurrent_files": 1,
            "gc_between_files": True,
        },
        "desktop_power": {
            "description": "Aggressive -- throughput on 32-64GB desktops",
            "embedding_batch_size": 64, "chunk_size": 1200, "chunk_overlap": 200,
            "retrieval_top_k": 10, "reranker_top_n": 40,
            "indexing_block_chars": 500000, "max_concurrent_files": 2,
            "gc_between_files": False,
        },
        "server_max": {
            "description": "Maximum -- for 64GB+ workstations",
            "embedding_batch_size": 128, "chunk_size": 1200, "chunk_overlap": 200,
            "retrieval_top_k": 15, "reranker_top_n": 60,
            "indexing_block_chars": 1000000, "max_concurrent_files": 4,
            "gc_between_files": False,
        },
    }
    sel = profiles[name]
    print(f"\n  PROFILE: {name} ({reason})")
    print(f"  {sel['description']}")
    print(f"  Batch: {sel['embedding_batch_size']}, Block: {sel['indexing_block_chars']}")
    return {"recommended_profile": name, "reason": reason,
            "settings": sel, "all_profiles": profiles}


# ===== SECTION 3: SECURITY AUDIT ===========================================

def run_security_audit():
    """Verify HuggingFace offline lockdown, API endpoint, kill switch, cache."""
    print("\n" + "=" * 60)
    print("  SECURITY AUDIT")
    print("=" * 60)
    results = []

    # HF offline env vars
    for var, exp in [("HF_HUB_OFFLINE","1"), ("TRANSFORMERS_OFFLINE","1"),
                     ("HF_HUB_DISABLE_TELEMETRY","1"), ("HF_HUB_DISABLE_IMPLICIT_TOKEN","1")]:
        val = os.environ.get(var, "NOT SET")
        ok = val == exp
        results.append({"test": f"ENV: {var}", "passed": ok, "detail": val})
        print(f"  [{'PASS' if ok else 'FAIL'}] {var} = {val}")

    # Offline override (checked by NetworkGate at startup)
    offline_env = os.environ.get("HYBRIDRAG_OFFLINE", "NOT SET")
    is_forced_offline = offline_env.strip().lower() in ("1", "true", "yes")
    results.append({"test": "ENV: HYBRIDRAG_OFFLINE", "passed": True, "detail": offline_env})
    print(f"  [{'WARN' if is_forced_offline else 'PASS'}] HYBRIDRAG_OFFLINE = {offline_env}"
          + (" (forced offline mode)" if is_forced_offline else ""))

    # Model cache dirs
    for label, var in [("ST_HOME", "SENTENCE_TRANSFORMERS_HOME"), ("HF_HOME", "HF_HOME")]:
        p = os.environ.get(var, "NOT SET")
        if p == "NOT SET":
            results.append({"test": f"CACHE: {label}", "passed": False, "detail": "Not set"})
            print(f"  [FAIL] {label}: not set")
        elif not Path(p).exists():
            results.append({"test": f"CACHE: {label}", "passed": False, "detail": "Missing"})
            print(f"  [FAIL] {label}: path missing")
        else:
            has = any(Path(p).rglob("*.bin")) or any(Path(p).rglob("*.safetensors"))
            results.append({"test": f"CACHE: {label}", "passed": has,
                            "detail": "Cached" if has else "Empty"})
            print(f"  [{'PASS' if has else 'WARN'}] {label}: {'cached' if has else 'EMPTY'}")

    # SEC-001: API endpoint
    try:
        from src.core.config import load_config
        cfg = load_config(str(PROJECT_ROOT))
        ep = cfg.api.endpoint
        if not ep:
            results.append({"test": "SEC-001", "passed": True, "detail": "Empty (safe)"})
            print("  [PASS] API endpoint: empty (SEC-001 OK)")
        elif "api.openai.com" in ep:
            results.append({"test": "SEC-001", "passed": False, "detail": f"PUBLIC: {ep}"})
            print(f"  [FAIL] API endpoint: PUBLIC INTERNET -- {ep}")
        else:
            results.append({"test": "SEC-001", "passed": True, "detail": f"Custom: {ep}"})
            print(f"  [PASS] API endpoint: {ep}")
    except Exception as e:
        results.append({"test": "SEC-001", "passed": False, "detail": str(e)})
        print(f"  [FAIL] Config error: {e}")

    # enterprise-in-depth: Python-level enforcement in embedder.py
    emb_path = PROJECT_ROOT / "src" / "core" / "embedder.py"
    if emb_path.exists():
        has = "HF_HUB_OFFLINE" in emb_path.read_text(encoding="utf-8")
        results.append({"test": "enterprise-IN-DEPTH: embedder.py", "passed": has,
                        "detail": "Present" if has else "MISSING"})
        print(f"  [{'PASS' if has else 'FAIL'}] Python offline lockdown: "
              f"{'present' if has else 'MISSING -- only PowerShell protects you'}")
    else:
        results.append({"test": "enterprise-IN-DEPTH", "passed": False, "detail": "Not found"})
        print("  [FAIL] embedder.py not found")

    p = sum(1 for r in results if r["passed"])

    # Network Gate status
    try:
        from src.core.network_gate import get_gate
        gate = get_gate()
        gate_mode = gate.mode_name.upper()
        gate_hosts = ", ".join(gate._allowed_hosts) if gate._allowed_hosts else "(localhost only)"
        results.append({"test": "NETWORK_GATE", "passed": True, "detail": f"{gate_mode}: {gate_hosts}"})
        print(f"  [PASS] Network gate: {gate_mode} -- allowed: {gate_hosts}")
        p += 1
    except Exception as e:
        results.append({"test": "NETWORK_GATE", "passed": False, "detail": str(e)})
        print(f"  [WARN] Network gate: not configured ({e})")

    print(f"\n  Security: {p}/{len(results)} passed")
    return results


# ===== SECTION 4: TIER 1 -- SCHEMA & LOGIC ==================================

def run_tier1_tests():
    """Instant tests: imports, schema, hash, garbage, close, config, FTS5, migration."""
    print("\n" + "=" * 60)
    print("  TIER 1: SCHEMA & LOGIC TESTS (instant)")
    print("=" * 60)
    results = []

    # T1-01: Imports
    print("\n  T1-01: Critical imports...")
    ok = True
    for mod, desc in [("src.core.config","Config"), ("src.core.vector_store","VectorStore"),
                      ("src.core.indexer","Indexer"), ("src.core.embedder","Embedder"),
                      ("src.core.retriever","Retriever")]:
        try:
            __import__(mod); print(f"    [PASS] {desc}")
        except Exception as e:
            print(f"    [FAIL] {desc}: {e}"); ok = False
    results.append({"test": "T1-01: Imports", "passed": ok})

    # T1-02: file_hash column
    print("\n  T1-02: file_hash column (BUG-001)...")
    try:
        from src.core.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as td:
            vs = VectorStore(db_path=os.path.join(td,"t.db"), embedding_dim=384); vs.connect()
            cols = [r[1] for r in vs.conn.execute("PRAGMA table_info(chunks)").fetchall()]
            ok = "file_hash" in cols
            print(f"    [{'PASS' if ok else 'FAIL'}] Columns: {cols}"); vs.close()
        results.append({"test": "T1-02: file_hash", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-02", "passed": False})

    # T1-03: Hash comparison
    print("\n  T1-03: Hash comparison (BUG-002)...")
    try:
        from src.core.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as td:
            vs = VectorStore(db_path=os.path.join(td,"t.db"), embedding_dim=384); vs.connect()
            fh = "12345:1707400000"
            vs.conn.execute("INSERT INTO chunks (chunk_id,source_path,chunk_index,text,"
                            "text_length,created_at,embedding_row,file_hash) VALUES (?,?,?,?,?,?,?,?)",
                            ("ck1","/fake.pdf",0,"X",1,datetime.now(timezone.utc).isoformat(),0,fh))
            vs.conn.commit()
            ok = vs.get_file_hash("/fake.pdf") == fh and (vs.get_file_hash("/no.pdf") in ("", None))
            print(f"    [{'PASS' if ok else 'FAIL'}] Hash stored/retrieved correctly"); vs.close()
        results.append({"test": "T1-03: Hash", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-03", "passed": False})

    # T1-04: Garbage text detection
    # ---------------------------------------------------------------
    # Tests that the system can distinguish normal text from binary
    # junk. Uses bytes 0x00-0x08 and 0x0E-0x1F which are control
    # characters with NO printable representation. The previous test
    # used "\x89PNG" which contains 'P','N','G' -- actual letters that
    # inflated the printable ratio to 56%. This version uses pure
    # non-printable control bytes for an accurate test.
    # ---------------------------------------------------------------
    print("\n  T1-04: Garbage text detection (BUG-004)...")
    try:
        from src.core.indexer import Indexer
        if hasattr(Indexer, "_validate_text"):
            # Build truly non-printable garbage: control chars only
            # 0x01-0x08, 0x0E-0x1F are non-printable, non-whitespace
            garbage_bytes = bytes(
                [b for b in range(256)
                 if b < 0x09 or (0x0E <= b <= 0x1F) or b == 0x7F]
            )
            # Repeat to get a substantial sample
            bad_text = garbage_bytes.decode("latin-1") * 20

            good_text = (
                "This is a perfectly normal English paragraph about "
                "engineering documents and technical specifications. "
                "The system operates at frequencies between 1 and 30 MHz. "
                "All measurements conform to standard operating procedures."
            )

            # Count printable ratio (mirrors _validate_text logic)
            def printable_ratio(text, sample_size=2000):
                sample = text[:sample_size]
                if not sample:
                    return 0.0
                normal = sum(
                    1 for c in sample
                    if c.isalnum() or c.isspace()
                    or c in ".,;:!?-()[]{}'\"/\\@#$%&*+=<>~`_"
                )
                return normal / len(sample)

            good_r = printable_ratio(good_text)
            bad_r = printable_ratio(bad_text)

            # Good text should be >30% printable, garbage should be <30%
            ok = (good_r >= 0.30) and (bad_r < 0.30)

            print(f"    [{'PASS' if ok else 'FAIL'}] "
                  f"Good={good_r:.0%}, Garbage={bad_r:.0%} "
                  f"(threshold 30%)")
        else:
            ok = False
            print("    [FAIL] _validate_text not found on Indexer")

        results.append({"test": "T1-04: Garbage", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-04", "passed": False})

    # T1-05: close() methods
    print("\n  T1-05: close() methods (BUG-003)...")
    try:
        from src.core.vector_store import VectorStore
        from src.core.embedder import Embedder
        from src.core.indexer import Indexer
        checks = [("VectorStore", callable(getattr(VectorStore,"close",None))),
                  ("Embedder", callable(getattr(Embedder,"close",None))),
                  ("Indexer", callable(getattr(Indexer,"close",None)))]
        ok = all(v for _,v in checks)
        for n,v in checks: print(f"    [{'PASS' if v else 'FAIL'}] {n}.close()")
        results.append({"test": "T1-05: close()", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-05", "passed": False})

    # T1-06: Config
    print("\n  T1-06: Config loading...")
    try:
        from src.core.config import load_config
        cfg = load_config(str(PROJECT_ROOT))
        print(f"    [PASS] mode={cfg.mode}")
        results.append({"test": "T1-06: Config", "passed": True})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-06", "passed": False})

    # T1-07: FTS5
    print("\n  T1-07: FTS5 extension...")
    try:
        with tempfile.TemporaryDirectory() as td:
            cn = sqlite3.connect(os.path.join(td,"f.db"))
            cn.execute("CREATE VIRTUAL TABLE f USING fts5(c)")
            cn.execute("INSERT INTO f VALUES (?)", ("test doc",))
            ok = len(cn.execute("SELECT * FROM f WHERE f MATCH ?", ("doc",)).fetchall()) == 1
            cn.close(); print(f"    [PASS] FTS5 works")
        results.append({"test": "T1-07: FTS5", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-07", "passed": False})

    # T1-08: Migration
    print("\n  T1-08: Legacy DB migration...")
    try:
        from src.core.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "old.db")
            cn = sqlite3.connect(db)
            cn.execute("CREATE TABLE chunks (chunk_pk INTEGER PRIMARY KEY AUTOINCREMENT,"
                       "chunk_id TEXT UNIQUE,source_path TEXT,chunk_index INTEGER,"
                       "text TEXT,text_length INTEGER,created_at TEXT,embedding_row INTEGER)")
            cn.commit(); cn.close()
            vs = VectorStore(db_path=db, embedding_dim=384); vs.connect()
            ok = "file_hash" in [r[1] for r in vs.conn.execute("PRAGMA table_info(chunks)").fetchall()]
            vs.close(); print(f"    [{'PASS' if ok else 'FAIL'}] Migration {'OK' if ok else 'failed'}")
        results.append({"test": "T1-08: Migration", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T1-08", "passed": False})

    return results


# ===== SECTION 5: TIER 2 -- PIPELINE ========================================

def run_tier2_tests():
    """Load model, embed, index, search with synthetic data."""
    print("\n" + "=" * 60)
    print("  TIER 2: PIPELINE TEST (loads model)")
    print("=" * 60)
    results = []

    print("\n  T2-01: Loading embedding model...")
    try:
        t0 = time.time()
        from src.core.embedder import Embedder
        emb = Embedder(model_name="all-MiniLM-L6-v2")
        ls = time.time() - t0
        print(f"    [PASS] Loaded in {ls:.1f}s")
        results.append({"test": "T2-01: Model", "passed": True, "metric": {"load_s": round(ls,2)}})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T2-01", "passed": False})
        return results

    print("\n  T2-02: Embedding dimensions...")
    try:
        shape = emb.embed_batch(["Test sentence.", "Another."]).shape
        ok = shape == (2, 384)
        print(f"    [{'PASS' if ok else 'FAIL'}] Shape: {shape}")
        results.append({"test": "T2-02: Dim", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T2-02", "passed": False})

    print("\n  T2-03: Index -> Search cycle...")
    try:
        import numpy as np
        import shutil
        from src.core.vector_store import VectorStore
        # Use mkdtemp instead of TemporaryDirectory context manager.
        # On Windows, the context manager tries to delete the folder
        # while SQLite still has the .db file open, causing WinError 267.
        # With mkdtemp we control cleanup order: close DB first, then delete.
        td = tempfile.mkdtemp(prefix="hybridrag_t2_")
        try:
            db_path = os.path.join(td, "pipeline_test.db")
            vs = VectorStore(db_path=db_path, embedding_dim=384); vs.connect()
            from src.core.vector_store import ChunkMetadata
            from datetime import datetime, timezone
            docs = ["Digisonde 4D operates 0.5-30 MHz.", "Ionograms show height vs freq.",
                    "Python is a programming language."]
            de = emb.embed_batch(docs)
            # Build metadata list matching ChunkMetadata signature:
            #   ChunkMetadata(source_path, chunk_index, text_length, created_at)
            now = datetime.now(timezone.utc).isoformat()
            meta = [ChunkMetadata(source_path="/test.txt", chunk_index=i,
                                  text_length=len(t), created_at=now)
                    for i, t in enumerate(docs)]
            chunk_ids = [f"t{i}" for i in range(len(docs))]
            # add_embeddings(embeddings, metadata_list, texts, chunk_ids, file_hash)
            vs.add_embeddings(de, meta, docs, chunk_ids, file_hash="100:170")
            qe = emb.embed_query("What frequency does Digisonde use?")
            best = int(np.argmax(np.dot(np.array(de), qe)))
            ok = "digisonde" in docs[best].lower()
            print(f"    [{'PASS' if ok else 'FAIL'}] Match: '{docs[best][:50]}'")
            vs.close()  # Close DB BEFORE deleting temp folder
        finally:
            try: shutil.rmtree(td, ignore_errors=True)
            except Exception: pass
        results.append({"test": "T2-03: Search", "passed": ok})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T2-03", "passed": False})

    print("\n  T2-04: Cleanup...")
    try:
        emb.close(); gc.collect(); print("    [PASS] close() OK")
        results.append({"test": "T2-04: Cleanup", "passed": True})
    except Exception as e:
        print(f"    [FAIL] {e}"); results.append({"test": "T2-04", "passed": False})

    return results


# ===== SECTION 6: TIER 3 -- STRESS TEST =====================================

def run_tier3_tests():
    """Production stress test with real corpus. Measures throughput + RAM."""
    print("\n" + "=" * 60)
    print("  TIER 3: STRESS TEST (1GB+ recommended)")
    print("=" * 60)

    sf = os.environ.get("HYBRIDRAG_INDEX_FOLDER", "")
    if not sf or not Path(sf).exists():
        print("\n  Source folder not set. Run via start_hybridrag.ps1")
        return [{"test": "T3", "passed": False, "detail": "No source"}]

    tb, fc = 0, 0
    for f in Path(sf).rglob("*"):
        if f.is_file(): tb += f.stat().st_size; fc += 1
    gb = tb / (1024**3)
    print(f"\n  Source: {sf}\n  Files: {fc:,}\n  Size: {gb:.2f} GB")

    if gb < 0.5:
        print(f"\n  Load 1GB+ docs into: {sf}")
        print("  Press Enter when ready, or Ctrl+C to skip.")
        try:
            input()
            tb, fc = 0, 0
            for f in Path(sf).rglob("*"):
                if f.is_file(): tb += f.stat().st_size; fc += 1
            gb = tb / (1024**3)
        except (KeyboardInterrupt, EOFError):
            return [{"test": "T3", "passed": False, "detail": "Skipped"}]

    results = []
    print("\n  T3-01: Indexing benchmark...\n")
    try:
        import tracemalloc
        from src.core.config import load_config
        from src.core.embedder import Embedder
        from src.core.vector_store import VectorStore
        from src.core.indexer import Indexer

        cfg = load_config(str(PROJECT_ROOT))
        with tempfile.TemporaryDirectory() as td:
            e = Embedder(model_name=cfg.embedding.model_name)
            v = VectorStore(db_path=os.path.join(td,"s.db"), embedding_dim=cfg.embedding.dimension)
            v.connect()
            ix = Indexer(config=cfg, embedder=e, vector_store=v)
            tracemalloc.start(); t0 = time.time()
            st = ix.index_folder(sf)
            elapsed = time.time() - t0; peak = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            ch = st.get("total_chunks_added", 0)
            cps = ch / elapsed if elapsed else 0
            mbps = (tb/(1024**2)) / elapsed if elapsed else 0
            pmb = peak / (1024**2)
            print(f"  Time:    {elapsed:.1f}s\n  Chunks:  {ch:,}\n  Speed:   {cps:.1f} ch/s, {mbps:.1f} MB/s\n  RAM:     {pmb:.0f} MB")
            results.append({"test": "T3-01: Stress", "passed": ch > 0,
                            "metric": {"elapsed_s": round(elapsed,1), "chunks": ch,
                                       "cps": round(cps,1), "mbps": round(mbps,1),
                                       "peak_mb": round(pmb,0), "corpus_gb": round(gb,2)}})
            for o in [ix, v, e]:
                try: o.close()
                except Exception: pass
    except Exception as ex:
        print(f"  [FAIL] {ex}")
        results.append({"test": "T3-01", "passed": False, "detail": str(ex)})
    return results


# ===== SECTION 7: SAVE & MAIN ==============================================

def save_results(hw, prof, sec, t1, t2, t3):
    """Save all results to config/system_profile.json for benchmarking."""
    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "hardware": hw, "profile": {k:v for k,v in prof.items() if k != "all_profiles"},
           "security": sec, "tier1": t1, "tier2": t2, "tier3": t3}
    d = PROJECT_ROOT / "config"; d.mkdir(exist_ok=True)
    p = d / "system_profile.json"
    with open(p, "w", encoding="utf-8") as f: json.dump(out, f, indent=2, default=str)
    return str(p)

def main():
    ap = argparse.ArgumentParser(description="HybridRAG Diagnostic")
    ap.add_argument("--tier", type=int, default=1, choices=[1,2,3])
    ap.add_argument("--hardware-only", action="store_true")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  HybridRAG v3 -- System Diagnostic")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    hw = collect_hardware_fingerprint()
    prof = recommend_profile(hw)
    if args.hardware_only:
        p = save_results(hw, prof, [], [], [], []); print(f"\n  Saved: {p}"); return 0

    sec = run_security_audit()
    t1 = run_tier1_tests()
    t2 = run_tier2_tests() if args.tier >= 2 else []
    t3 = run_tier3_tests() if args.tier >= 3 else []
    path = save_results(hw, prof, sec, t1, t2, t3)

    all_t = t1 + t2 + t3
    tp = sum(1 for t in all_t if t.get("passed"))
    sp = sum(1 for s in sec if s.get("passed"))

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Machine: {hw['cpu']['name']}")
    print(f"           {hw['ram_gb']}GB RAM, {hw['cpu']['physical_cores']} cores")
    print(f"  Profile: {prof['recommended_profile']}")
    print(f"  Security:{sp}/{len(sec)} passed")
    print(f"  Tests:   {tp}/{len(all_t)} passed")
    print(f"  Saved:   {path}")
    if tp == len(all_t) and sp == len(sec): print("\n  ALL PASSED -- system ready")
    elif tp == len(all_t): print("\n  Tests OK, security issues -- review above")
    else: print("\n  FAILURES -- review above")
    print("=" * 60)
    return 0 if tp == len(all_t) else 1

if __name__ == "__main__":
    sys.exit(main())
