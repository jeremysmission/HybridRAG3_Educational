"""
HybridRAG3 -- Rebuild master_toolkit.ps1
FILE: tools/_rebuild_toolkit.py

WHAT THIS DOES (single pass, no patches needed):
  1. Reads master_toolkit.ps1
  2. Extracts all 12 Python here-string blocks into tools/py/*.py
  3. Rewrites master_toolkit.ps1 with external python calls (zero here-strings)
  4. Removes the Run-TempPython helper function (no longer needed)
  5. Fixes ALL known PS 5.1 syntax issues:
     - Non-ASCII characters (em-dashes, smart quotes, etc)
     - & in double-quoted strings
     - dot-backslash path references in double-quoted strings
     - Trailing backslash in double-quoted strings
     - ${var}UNIT collisions (${size}KB)
     - ($var WORD) parenthesized variable+text in strings
  6. Ensures UTF-8 BOM + CRLF line endings throughout

WHY:
  PowerShell 5.1 has a parser bug where it fails to parse large scripts
  containing here-strings with complex content (curly braces, f-strings).
  Additionally, PS 5.1 has strict rules about special characters inside
  double-quoted strings that differ from PS 7+.

  Instead of patching these issues one by one, this script redesigns the
  toolkit so Python lives in separate .py files and PS syntax is correct
  from the start.

USAGE: python tools/_rebuild_toolkit.py
"""
import os
import sys
import re


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ROOT = os.getcwd()
TOOLKIT_PATH = os.path.join(ROOT, "tools", "master_toolkit.ps1")
PY_DIR = os.path.join(ROOT, "tools", "py")

# Map PS function names to Python filenames
NAME_MAP = {
    "rag-store-key": "store_key",
    "rag-store-endpoint": "store_endpoint",
    "rag-show-creds": "show_creds",
    "rag-clear-creds": "clear_creds",
    "rag-debug-url": "debug_url",
    "rag-test-api-verbose": "test_api_verbose",
    "rag-ollama-test": "ollama_test",
    "rag-index-status": "index_status",
    "rag-index-reset": "index_reset",
    "rag-net-check": "net_check",
    "rag-ssl-check": "ssl_check",
    "rag-versions": "versions",
}

# Non-ASCII replacements -- every known character we have encountered
UNICODE_REPLACEMENTS = {
    "\u2014": "--",   # em dash
    "\u2013": "--",   # en dash
    "\u2018": "'",    # left single smart quote
    "\u2019": "'",    # right single smart quote
    "\u201c": '"',    # left double smart quote
    "\u201d": '"',    # right double smart quote
    "\u2713": "[OK]",
    "\u2717": "[FAIL]",
    "\u26a0": "[WARN]",
    "\u25cb": "[SKIP]",
    "\u2192": "->",
    "\u2500": "-",
    "\u2588": "#",
    "\u25b6": ">",
    "\u25c0": "<",
    "\u2550": "=",
    "\u2591": ".",
}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def fix_non_ascii_text(text):
    """Replace non-ASCII characters with ASCII equivalents.
    
    Handles double-encoded UTF-8 first (appears as multi-char sequences
    like 'a euro right-quote' when decoded), then individual characters.
    Double-encoded must be fixed FIRST or the individual char replacement
    will turn the right-quote into a real " and corrupt PS strings.
    """
    # --- Double-encoded sequences (as decoded text) FIRST ---
    # Double-encoded em-dash appears as: \u00e2\u20ac\u201d  (a + euro + right-dq)
    text = text.replace("\u00e2\u20ac\u201d", "--")
    # Double-encoded en-dash appears as: \u00e2\u20ac\u201c  (a + euro + left-dq)
    text = text.replace("\u00e2\u20ac\u201c", "--")
    # Also handle partial double-encoding: \u00e2\u20ac + other
    text = text.replace("\u00e2\u20ac", "-")
    
    # --- Individual character replacements ---
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def fix_non_ascii_bytes(raw):
    """Strip non-ASCII bytes, handling both single and double-encoded UTF-8.
    
    the developer's backup has DOUBLE-ENCODED em-dashes:
      Original em-dash U+2014 = UTF-8 E2 80 94
      Re-saved as Win-1252 then re-encoded to UTF-8:
        E2->C3 A2, 80->E2 82 AC, 94->E2 80 9D
    
    If we only handle single-encoded sequences, the double-encoded ones
    produce garbage like inserting a literal " into a PS string, which
    destroys the entire file's parsing.
    """
    # --- Double-encoded sequences FIRST (longer patterns match first) ---
    # Double-encoded em-dash: \xc3\xa2\xe2\x82\xac\xe2\x80\x9d
    result = raw.replace(b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9d", b"--")
    # Double-encoded en-dash: \xc3\xa2\xe2\x82\xac\xe2\x80\x9c
    result = result.replace(b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9c", b"--")
    
    # --- Single-encoded sequences ---
    result = result.replace(b"\xe2\x80\x94", b"--")   # em dash
    result = result.replace(b"\xe2\x80\x93", b"--")   # en dash
    result = result.replace(b"\xe2\x80\x98", b"'")    # left smart quote
    result = result.replace(b"\xe2\x80\x99", b"'")    # right smart quote
    result = result.replace(b"\xe2\x80\x9c", b'"')    # left double smart quote
    result = result.replace(b"\xe2\x80\x9d", b'"')    # right double smart quote
    
    # --- Brute force: any remaining byte > 127 becomes a hyphen ---
    result = bytes(b if b < 128 else ord("-") for b in result)
    return result


def extract_python_blocks(lines):
    """
    Walk through the PS1 file and extract every Python here-string block.

    Returns a list of dicts:
      func:   name of the PS function containing the block
      start:  line index of the $pythonCode = @' line
      end:    line index of the closing '@ line
      type:   'single' or 'double' (which quote style)
      code:   list of Python source lines (strings)
    """
    blocks = []
    in_hs = False
    hs_type = None
    hs_start = 0
    current_func = None
    python_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track which function we are inside
        if stripped.startswith("function ") and "{" in stripped:
            current_func = stripped.split()[1].split("{")[0].strip()

        # Detect here-string opening: $pythonCode = @' or $pythonCode = @"
        if "= @'" in stripped or '= @"' in stripped:
            hs_type = "single" if "= @'" in stripped else "double"
            hs_start = i
            in_hs = True
            python_lines = []
            continue

        # Detect here-string closing: '@ or "@
        if in_hs and (stripped == "'@" or stripped == '"@'):
            blocks.append({
                "func": current_func,
                "start": hs_start,
                "end": i,
                "type": hs_type,
                "code": python_lines[:],
            })
            in_hs = False
            continue

        # Accumulate Python lines inside the here-string
        if in_hs:
            python_lines.append(line.rstrip("\r\n"))

    return blocks


def build_py_file(func_name, code_lines):
    """
    Create standalone Python file content from extracted code.

    For store_key and store_endpoint, the PS wrapper passes the user's
    input as a command-line argument, so the Python code uses sys.argv[1]
    instead of the PS variable ($key / $endpoint).
    """
    py_name = NAME_MAP.get(func_name, func_name)

    # --- Special case: rag-store-key ---
    # Original used $key (PS variable expansion in double-quoted here-string)
    # New version takes the key as sys.argv[1]
    if func_name == "rag-store-key":
        code = (
            'import sys\n'
            'import keyring\n'
            '\n'
            'key = sys.argv[1]\n'
            'keyring.set_password("hybridrag", "azure_api_key", key)\n'
            'stored = keyring.get_password("hybridrag", "azure_api_key")\n'
            'if stored == key:\n'
            '    print("  [OK] API key stored successfully.")\n'
            '    print("  Preview: " + stored[:4] + "..." + stored[-4:])\n'
            'else:\n'
            '    print("  [ERROR] Key storage failed. Keyring may not be available.")\n'
        )
        return py_name, code

    # --- Special case: rag-store-endpoint ---
    if func_name == "rag-store-endpoint":
        code = (
            'import sys\n'
            'import keyring\n'
            '\n'
            'endpoint = sys.argv[1]\n'
            'keyring.set_password("hybridrag", "azure_endpoint", endpoint)\n'
            'stored = keyring.get_password("hybridrag", "azure_endpoint")\n'
            'if stored == endpoint:\n'
            '    print("  [OK] Endpoint stored successfully.")\n'
            '    print("  Value: " + stored)\n'
            'else:\n'
            '    print("  [ERROR] Endpoint storage failed.")\n'
        )
        return py_name, code

    # --- All other functions: use extracted code as-is ---
    code = "\n".join(code_lines)
    code = fix_non_ascii_text(code)
    return py_name, code


def replace_herestrings_with_calls(lines, blocks):
    """
    Replace each here-string block + Run-TempPython call with a single
    python "tools/py/name.py" line.

    Preserves all surrounding PS code (prompts, validation, Write-Host).
    """
    # Build skip ranges: from $pythonCode = @' through Run-TempPython line
    skip_ranges = []
    for b in blocks:
        assign_line = b["start"]
        # Scan forward from closing '@ to find Run-TempPython
        run_line = b["end"]
        for j in range(b["end"] + 1, min(b["end"] + 5, len(lines))):
            if "Run-TempPython" in lines[j]:
                run_line = j
                break
        skip_ranges.append((assign_line, run_line, b["func"]))

    skip_ranges.sort(key=lambda x: x[0])

    new_lines = []
    for i, line in enumerate(lines):
        in_skip = False
        for start, end, func in skip_ranges:
            if start <= i <= end:
                in_skip = True
                if i == start:
                    # Insert the replacement call
                    py_name = NAME_MAP.get(func, func)
                    indent = "    "
                    if func == "rag-store-key":
                        new_lines.append(
                            indent + 'python "tools/py/store_key.py" $key\r\n'
                        )
                    elif func == "rag-store-endpoint":
                        new_lines.append(
                            indent + 'python "tools/py/store_endpoint.py" $endpoint\r\n'
                        )
                    else:
                        new_lines.append(
                            indent + 'python "tools/py/%s.py"\r\n' % py_name
                        )
                break
        if not in_skip:
            new_lines.append(line)

    return new_lines


def remove_run_temppython(lines):
    """Remove the Run-TempPython helper function entirely."""
    result = []
    skip_func = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()
        if "function Run-TempPython" in stripped:
            skip_func = True
            brace_depth = 0
        if skip_func:
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0 and stripped == "}":
                skip_func = False
                continue
            continue
        result.append(line)

    return result


def fix_ps_syntax(lines):
    """
    Fix all known PS 5.1 syntax issues in PowerShell code lines.

    These are issues that exist in the original PS code (outside of
    here-strings) and would cause parse errors in PS 5.1.

    Strategy: build complex display strings with + concatenation in a
    variable, then pass to Write-Host. This is immune to PS parser
    ambiguity because + concatenation has no special parsing rules.

    The -f format operator does NOT work reliably in PS 5.1 because
    {0} placeholders get parsed as script blocks inside expressions.
    """
    result = []

    for line in lines:
        # --- Fix: & in double-quoted strings ---
        # PS 5.1 reserves & for future use inside double-quoted strings
        # "API & Model..." -> "API and Model..."
        if '"API & Model' in line:
            line = line.replace(
                '"API & Model Environment Variables"',
                '"API and Model Environment Variables"'
            )

        # --- Fix: dot-backslash path in double-quoted strings ---
        # PS interprets .\ as a member reference operator
        # ".\temp_diag" -> '.\temp_diag'  (single quotes = literal)
        if '".\\temp_diag"' in line:
            line = line.replace('".\\temp_diag"', "'.\\temp_diag'")

        # --- Fix: trailing backslash before closing double-quote ---
        # The \ escapes the " so PS never sees the string end
        # "  Removed: temp_diag\" -> '  Removed: temp_diag\'
        if 'Removed: temp_diag\\"' in line:
            line = line.replace(
                'Write-Host "  Removed: temp_diag\\"',
                "Write-Host '  Removed: temp_diag\\'"
            )

        # --- Fix: ${var}UNIT collisions ---
        # PS treats KB/MB/GB after ${var} as numeric multiplier
        # Also: ($var UNIT) inside double-quoted strings starts subexpression
        # FIX: build the string with + concatenation in a separate variable
        if '${size}KB' in line or '($($size) KB)' in line:
            line = (
                '        $display = "    " + $b.Name + " (" + $size + " KB)"\r\n'
                '        Write-Host $display -ForegroundColor Gray\r\n'
            )

        # --- Fix: ($var WORD) in double-quoted strings ---
        if '$backupPath ($sizeMB MB)' in line:
            line = (
                '    $display = "  [OK] Backup created: " + $backupPath + " (" + $sizeMB + " MB)"\r\n'
                '    Write-Host $display -ForegroundColor Green\r\n'
            )

        result.append(line)

    return result


def write_ps1(path, lines):
    """Write PS1 file with UTF-8 BOM and CRLF line endings.
    
    Also does a final brute-force pass to strip any remaining non-ASCII
    bytes that slipped through the text-level replacement map.
    """
    raw = b"\xef\xbb\xbf"  # UTF-8 BOM (required by PS 5.1)
    for line in lines:
        clean = line.rstrip("\r\n")
        raw += (clean + "\r\n").encode("utf-8")
    
    # Brute-force: strip any non-ASCII bytes (except the BOM we just added)
    bom = raw[:3]
    body = raw[3:]
    body = fix_non_ascii_bytes(body)
    raw = bom + body
    
    with open(path, "wb") as f:
        f.write(raw)


def write_py_files(blocks):
    """Write all extracted Python files to tools/py/."""
    os.makedirs(PY_DIR, exist_ok=True)

    for b in blocks:
        py_name, code = build_py_file(b["func"], b["code"])
        path = os.path.join(PY_DIR, py_name + ".py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        print("  [OK] tools/py/%s.py (%d bytes)" % (py_name, len(code)))


def verify_py_files():
    """Check all .py files for non-ASCII and fix if needed."""
    all_clean = True
    for fname in sorted(os.listdir(PY_DIR)):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(PY_DIR, fname)
        raw = open(fpath, "rb").read()
        cleaned = fix_non_ascii_bytes(raw)
        if cleaned != raw:
            open(fpath, "wb").write(cleaned)
            print("  [FIXED] %s (non-ASCII removed)" % fname)
            all_clean = False
    if all_clean:
        print("  [OK] All .py files are ASCII-clean")


def verify_ps1(path):
    """Run verification checks on the final PS1 file."""
    data = open(path, "rb").read()
    text = data.decode("utf-8-sig")
    lines = text.split("\n")

    # Line ending check
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    print("  [%s] Line endings: CRLF=%d, LF=%d" % (
        "OK" if lf_only == 0 else "WARN", crlf, lf_only))

    # Here-string check
    hs_count = sum(1 for l in lines if "= @'" in l or '= @"' in l)
    print("  [%s] Here-strings remaining: %d" % (
        "OK" if hs_count == 0 else "FAIL", hs_count))

    # Non-ASCII check (excluding BOM)
    non_ascii = [ch for ch in text if ord(ch) > 127]
    print("  [%s] Non-ASCII characters: %d" % (
        "OK" if len(non_ascii) == 0 else "WARN", len(non_ascii)))

    # Known PS 5.1 problem patterns
    problems = []
    for i, line in enumerate(lines, 1):
        if '"API & ' in line:
            problems.append("Line %d: & in double-quoted string" % i)
        if '${size}KB' in line:
            problems.append("Line %d: ${size}KB collision" % i)
        if '$backupPath ($sizeMB' in line:
            problems.append("Line %d: ($sizeMB in string" % i)
        if '".\\temp' in line:
            problems.append("Line %d: dot-backslash in double-quoted string" % i)

    if problems:
        for p in problems:
            print("  [WARN] %s" % p)
    else:
        print("  [OK] No known PS 5.1 syntax issues")

    # Run-TempPython check
    has_rtp = "Run-TempPython" in text
    print("  [%s] Run-TempPython: %s" % (
        "OK" if not has_rtp else "WARN",
        "removed" if not has_rtp else "still present"))

    # Function count
    func_count = sum(1 for l in lines if l.strip().startswith("function "))
    print("  [OK] Functions: %d" % func_count)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print()
    print("  HybridRAG3 Master Toolkit Rebuild")
    print("  ==================================")
    print()

    if not os.path.exists(TOOLKIT_PATH):
        print("  [ERROR] %s not found. Run from project root." % TOOLKIT_PATH)
        sys.exit(1)

    # Step 1: Read original
    with open(TOOLKIT_PATH, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    print("  Read %d lines from master_toolkit.ps1" % len(lines))

    # Step 2: Extract Python blocks
    blocks = extract_python_blocks(lines)
    print("  Found %d Python blocks to extract" % len(blocks))
    print()

    # Step 3: Write Python files
    write_py_files(blocks)
    print()

    # Step 4: Verify Python files (fix non-ASCII)
    verify_py_files()
    print()

    # Step 5: Backup original
    backup_path = TOOLKIT_PATH + ".pre_extract.bak"
    with open(backup_path, "wb") as f:
        f.write(open(TOOLKIT_PATH, "rb").read())
    print("  [OK] Backup: %s" % os.path.basename(backup_path))

    # Step 6: Replace here-strings with external python calls
    lines = replace_herestrings_with_calls(lines, blocks)

    # Step 7: Remove Run-TempPython function
    lines = remove_run_temppython(lines)

    # Step 8: Fix all known PS 5.1 syntax issues
    lines = fix_ps_syntax(lines)

    # Step 9: Fix non-ASCII in PS lines
    lines = [fix_non_ascii_text(line) for line in lines]

    # Step 10: Write final file with BOM + CRLF
    write_ps1(TOOLKIT_PATH, lines)
    print("  [OK] master_toolkit.ps1 written: %d lines" % len(lines))
    print()

    # Step 11: Verify everything
    print("  --- Verification ---")
    verify_ps1(TOOLKIT_PATH)
    verify_py_files()

    print()
    print("  DONE. Test with:")
    print("    . .\\tools\\master_toolkit.ps1")
    print("    rag-help")
    print()


if __name__ == "__main__":
    main()
