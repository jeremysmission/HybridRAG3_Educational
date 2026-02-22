# Permissions
Full read/write/edit/create/delete/run permission. No confirmation needed.
Only confirm before: deleting project root, pushing to GitHub remote.
Web/forum research permitted without asking.

# Standing Rules
- No patches, only redesigns
- Classes under 500 lines
- No hardcoded paths
- No non-ASCII characters in scripts
- No inline Python in PowerShell functions
- PowerShell: UTF-8 BOM, CRLF, one command per code block, only @'...'@ here-strings
- Log output tags: [OK] [FAIL] [WARN] -- never em-dashes
- Zero-trust offline-default architecture
- Never commit API keys, secrets, model cache files, or zips (except releases/)

# Git Rules
<!-- detailed: docs/GIT_REPO_RULES.md -->
- Commit/push only from home machine
- Run sanitization check (docs/GIT_REPO_RULES.md) before every commit
- Banned words in repo: see docs/GIT_REPO_RULES.md for full list
- deploy_comments.ps1 and .AI assistant/ are intentionally untracked
- Sync to educational: tools/sync_to_educational.py (one-way, sanitized)
- Skip patterns: *.docx, *.xlsx, ~$* (binary Office files)

# Paths
- Source data: {SOURCE_DIR}
- Indexed data: {DATA_DIR}
- DialedIn clone: {PROJECT_ROOT}_DialedIn
- Educational clone: separate repo, never direct-push from here

# Banned Models (NDAA / regulatory)
<!-- detailed: docs/MODEL_AUDIT.md -->
- NO China-origin: Qwen/Alibaba, DeepSeek, BGE/BAAI
- NO Meta/Llama (regulatory ban)
- Approved stack: phi4-mini, mistral:7b, phi4:14b-q4_K_M, gemma3:4b, mistral-nemo:12b
- Model definitions: scripts/_model_meta.py, scripts/_set_model.py
- Default offline model: phi4-mini (config/default_config.yaml)

# Eval Protection
- NEVER modify: scripts/run_eval.py, tools/eval_runner.py, tools/score_results.py, tools/run_all.py, Eval/*.json
- NEVER enable reranker for multi-type eval (destroys unanswerable/injection/ambiguous scores)
- Injection trap: AES_RE regex catches "AES-512" anywhere in answer text

# API
- FastAPI server: src/api/server.py | routes: src/api/routes.py | models: src/api/models.py
- TestClient MUST use context manager: `with TestClient(app) as client:`
- Binds localhost only (127.0.0.1:8000)

# Windows Gotchas
- UnicodeEncodeError: wrap stdout with io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
- Chunker takes config object, not (chunk_size, overlap) args
- ensure_directories is module-level function, not PathsConfig method
