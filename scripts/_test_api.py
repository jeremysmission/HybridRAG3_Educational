# ============================================================================
# HybridRAG v3 - API Connectivity Test (scripts/_test_api.py)
# ============================================================================
#
# WHAT THIS FILE DOES:
#   Sends a tiny test question to your company's GPT API to verify that:
#     1. Your API key is valid (not expired or rejected)
#     2. The endpoint URL is reachable (you're on the right network)
#     3. The model actually responds with an answer
#
#   Think of it like pinging a website to see if it's alive, except
#   instead of a ping we send a real question ("Say hello in 5 words")
#   and check that we get a real answer back.
#
# WHO CALLS THIS:
#   api_mode_commands.ps1 -> rag-test-api function
#   You never need to run this file directly.
#
# WHAT IT PRINTS:
#   - Mode, API readiness, and endpoint URL
#   - The test question and the AI's answer
#   - How many tokens were used and estimated cost
#   - PASS or FAIL with troubleshooting hints
#
# INTERNET ACCESS: YES - makes one HTTP request to your API endpoint.
# ============================================================================

import sys
import time
import os

# Add the project root to Python's search path so we can import our modules.
# HYBRIDRAG_PROJECT_ROOT is set by start_hybridrag.ps1 when you launch HybridRAG.
sys.path.insert(0, os.environ.get('HYBRIDRAG_PROJECT_ROOT', '.'))

# Import the config loader (reads settings from config/default_config.yaml)
# and the LLM router (decides whether to talk to Ollama or the GPT API)
from src.core.config import load_config, ensure_directories
from src.core.llm_router import LLMRouter

# -- Step 1: Load config and force online mode for this test --
# Even if you're currently in offline mode, this test temporarily
# switches to online to test the API connection. It does NOT change
# your config file - the mode override only lasts for this script.
config = load_config('.')
ensure_directories(config)
config.mode = 'online'

# -- Step 2: Create the LLM router --
# The router will try to find your API key by checking:
#   1. Windows Credential Manager (keyring) - most secure
#   2. OPENAI_API_KEY environment variable - fallback
# If neither has a key, api_configured will be False.
router = LLMRouter(config)
status = router.get_status()

# -- Step 3: Print what we found --
print('  Mode:      ' + str(status['mode']))
print('  API ready: ' + str(status['api_configured']))
print('  Endpoint:  ' + str(status.get('api_endpoint', 'NOT SET')))
print()

# -- Step 4: Stop early if no API key was found --
if not status['api_configured']:
    print('  ERROR: API not configured. Run rag-store-key first.')
    sys.exit(1)

# -- Step 5: Send a tiny test question to the API --
# We use a simple prompt so the response is fast and cheap.
# "Say hello in exactly 5 words" should produce something like
# "Hello there, how are you?" - about 5-10 tokens of output.
print('  Sending test query: "Say hello in exactly 5 words."')
print('  Waiting for response...')
print()

# time.time() gives us the current time in seconds.
# We record the time before and after the API call to measure latency.
start = time.time()
resp = router.query('Say hello in exactly 5 words.')
elapsed = (time.time() - start) * 1000   # Convert seconds to milliseconds

# -- Step 6: Check the result --
if resp:
    # SUCCESS - the API responded with an answer
    print('  PASS  API responded successfully!')
    print('  Answer:     ' + resp.text.strip()[:100])   # Show first 100 chars
    print('  Model:      ' + resp.model)                 # Which model answered
    print('  Tokens in:  ' + str(resp.tokens_in))        # How many tokens we sent
    print('  Tokens out: ' + str(resp.tokens_out))       # How many tokens came back
    print('  Latency:    ' + str(round(elapsed)) + 'ms') # How long it took

    # Estimate the cost of this one query.
    # GPT-3.5 Turbo pricing: $0.0005 per 1K input tokens, $0.0015 per 1K output
    # A typical query costs about $0.002 (one fifth of a penny)
    cost = (resp.tokens_in * 0.0005 + resp.tokens_out * 0.0015) / 1000
    print('  Est. cost:  $' + format(cost, '.6f'))
    print()
    print('  API mode is ready for use!')
else:
    # FAILURE - the API did not respond
    # This could mean: wrong key, wrong URL, not on the right network,
    # or the company API server is down.
    print('  FAIL  API did not respond.')
    print('  Check: Is your endpoint URL correct?')
    print('  Check: Is your API key valid?')
    print('  Check: Are you on the intranet/VPN?')
    print('  Run: rag-cred-status')
    sys.exit(1)
