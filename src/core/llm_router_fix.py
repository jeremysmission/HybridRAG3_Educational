# ===========================================================================
# llm_router_fix.py -- Smart URL + Auth Header Router for HybridRAG v3
# ===========================================================================
#
# WHAT THIS FILE DOES (PLAIN ENGLISH):
#   When HybridRAG needs to ask an AI model a question (like "summarize
#   this document"), it has to send an HTTP request over the internet to
#   either Microsoft Azure or OpenAI. Think of it like mailing a letter:
#   you need the right address (URL) and the right ID badge (auth header).
#
#   The problem was:
#     BLOCKER-1: We were showing the wrong ID badge. Azure wants a badge
#                labeled "api-key" but we were showing one labeled "Bearer".
#                Azure's bouncer (the server) says "401 Unauthorized" which
#                means "I don't recognize your badge format."
#
#     BLOCKER-2: The address was getting doubled. Imagine writing
#                "123 Main St, Apt 4, 123 Main St, Apt 4" on an envelope.
#                The post office says "404 Not Found" -- that address
#                doesn't exist because the path is duplicated.
#
#   This file fixes both by:
#     1. Looking at the URL to figure out if it's Azure or OpenAI
#     2. Picking the correct badge (header) format automatically
#     3. Checking if the address is already complete before adding to it
#
# HOW IT FITS INTO HYBRIDRAG:
#   Your pipeline: User Question -> Embed -> Retrieve -> [THIS FILE] -> Answer
#   This file handles the LLM call. Everything else stays the same.
#
# INSTALLATION:
#   Option A (recommended): Run tools\write_llm_router_fix.ps1
#   Option B: Copy this file to src\core\ and import from it:
#       from src.core.llm_router_fix import call_llm_api
#
# RESEARCH:
#   - Microsoft Q&A: Azure requires "api-key" header, not "Bearer"
#   - GitHub openai/codex #3048: Custom APIM endpoints reject Bearer
#   - GitHub openai/codex #1192: Bearer token causes 401 on Azure
# ===========================================================================


# ---------------------------------------------------------------------------
# IMPORTS -- Loading tools this file needs
# ---------------------------------------------------------------------------
# "import" is like opening a toolbox. Each one gives us pre-built tools.

import requests   # Sends HTTP requests (like a web browser does behind the scenes)
import logging    # Writes diagnostic messages to a log file or console
import time       # Lets us measure how long operations take (stopwatch)


# ---------------------------------------------------------------------------
# LOGGER SETUP
# ---------------------------------------------------------------------------
# A "logger" is like a flight recorder in an aircraft. It writes messages
# about what the code is doing so you can review them later. __name__ is
# a special Python variable that equals this file's name, so log messages
# will be tagged with "llm_router_fix" as the source.

logger = logging.getLogger(__name__)


# ===========================================================================
# FUNCTION 1: detect_provider
# ===========================================================================
# WHAT IT DOES:
#   Looks at the API endpoint URL and figures out if it belongs to
#   Microsoft Azure or standard OpenAI. This is critical because they
#   use completely different authentication header formats.
#
# ANALOGY:
#   You're at an airport with two terminals. Terminal A (Azure) requires
#   a blue badge. Terminal B (OpenAI) requires a red badge. This function
#   reads your boarding pass (URL) and tells you which terminal to go to.
#
# INPUT:  endpoint_url -- a text string like "https://myco.openai.azure.com"
# OUTPUT: Returns either the text "azure" or the text "openai"
# ---------------------------------------------------------------------------

def detect_provider(endpoint_url):
    """Detect Azure vs OpenAI from the endpoint URL."""

    # .lower() converts everything to lowercase so our checks work
    # regardless of capitalization. "AZURE" and "azure" both match.
    url_lower = endpoint_url.lower()

    # Check patterns from most specific to least specific.
    # We check the most reliable pattern first.

    if ".openai.azure.com" in url_lower:
        # Standard Azure OpenAI format.
        # Example: https://mycompany.openai.azure.com
        return "azure"

    elif ".azure-api.net" in url_lower:
        # Azure API Management (APIM) gateway -- a custom domain that
        # some companies put in front of Azure OpenAI for extra control.
        # Example: https://api.mycompany.azure-api.net
        return "azure"

    elif "azure" in url_lower:
        # Catch-all: if "azure" appears anywhere, assume Azure.
        # Handles unusual or company-specific URL formats.
        return "azure"

    else:
        # No Azure patterns found. Assume standard OpenAI or a
        # compatible API (like Ollama running in OpenAI mode).
        return "openai"


# ===========================================================================
# FUNCTION 2: build_api_url
# ===========================================================================
# WHAT IT DOES:
#   Takes whatever URL the user stored and makes sure it is complete
#   and correct for the API call. Most importantly, it PREVENTS URL
#   DOUBLING (Blocker-2).
#
# THE DOUBLING PROBLEM:
#   Your IT gave you this full Postman URL:
#     https://myco.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-02-01
#
#   The old code assumed you only stored the base domain:
#     https://myco.openai.azure.com
#
#   So it appended the path AGAIN, creating:
#     https://myco.openai.azure.com/openai/deployments/.../openai/deployments/...
#
#   That doubled URL gives "404 Not Found" (no such address exists).
#
# THE FIX:
#   Check if "/chat/completions" is already in the URL.
#   YES -> URL is complete, use it exactly as stored.
#   NO  -> Figure out what's missing and add ONLY those parts.
#
# INPUT:
#   endpoint_url    -- The stored URL (full, partial, or base-only)
#   deployment_name -- Azure model name (default: gpt-35-turbo)
#                      Change this if IT named the deployment differently
#   api_version     -- Azure API version (default: 2024-02-01)
#                      Azure REQUIRES this parameter in every request
#
# OUTPUT: A complete URL string ready for the HTTP POST request
# ---------------------------------------------------------------------------

def build_api_url(endpoint_url, deployment_name="gpt-35-turbo", api_version="2024-02-01"):
    """Build the full API URL without doubling any path segments."""

    # .strip() removes invisible whitespace from both ends (spaces, tabs,
    # newlines that sneak in from copy-paste and break URLs silently).
    # .rstrip("/") removes any trailing slash to prevent double-slash
    # like "mycompany.com//openai" in the final URL.
    url = endpoint_url.strip().rstrip("/")

    # ----- ANTI-DOUBLING CHECK -----
    # If the URL already contains "/chat/completions", the user stored
    # the full Postman URL. Perfect -- use it exactly as given.
    # This one check prevents the entire doubling problem.
    if "/chat/completions" in url:
        logger.info("URL already has /chat/completions -- using as-is")
        return url

    # If we get here, the URL is incomplete. We need to add parts.
    # Azure and OpenAI have different URL structures:
    #   Azure:  /openai/deployments/{model}/chat/completions?api-version=...
    #   OpenAI: /v1/chat/completions
    provider = detect_provider(url)

    if provider == "azure":
        # ----- AZURE URL BUILDING -----
        if "/openai/deployments/" in url:
            # User stored a partial path like:
            #   https://myco.openai.azure.com/openai/deployments/gpt-35-turbo
            # Just add the endpoint and api-version.
            full_url = url + "/chat/completions"
            if "api-version" not in full_url:
                full_url = full_url + "?api-version=" + api_version
            return full_url
        else:
            # User stored only the base domain like:
            #   https://myco.openai.azure.com
            # Build the entire Azure path from scratch.
            return (
                url                                      # Base domain
                + "/openai/deployments/"                  # Azure-specific prefix
                + deployment_name                        # Model (e.g., gpt-35-turbo)
                + "/chat/completions"                    # API endpoint
                + "?api-version=" + api_version          # Required parameter
            )
    else:
        # ----- OPENAI URL BUILDING (much simpler) -----
        if "/v1/" in url:
            return url + "/chat/completions"
        else:
            return url + "/v1/chat/completions"


# ===========================================================================
# FUNCTION 3: build_headers
# ===========================================================================
# WHAT IT DOES:
#   Creates the HTTP headers (metadata on the envelope of every request)
#   using the CORRECT authentication format for the detected provider.
#
# WHY THIS IS THE KEY FIX:
#   HTTP headers are info on the OUTSIDE of the envelope. The server
#   reads them BEFORE looking at your question. Wrong header = instant
#   rejection with "401 Unauthorized".
#
#   Azure expects:  { "api-key": "abc123def456" }
#   OpenAI expects: { "Authorization": "Bearer abc123def456" }
#
#   Sending "Bearer" to Azure = instant 401. THIS WAS BLOCKER-1.
#
# INPUT:
#   endpoint_url -- The API URL (used to detect which provider)
#   api_key      -- The secret API key string from credential manager
#
# OUTPUT:
#   A Python dictionary of HTTP headers. Example:
#   {"Content-Type": "application/json", "api-key": "abc123"}
# ---------------------------------------------------------------------------

def build_headers(endpoint_url, api_key):
    """Build correct auth headers for the detected provider."""

    provider = detect_provider(endpoint_url)

    # Content-Type tells the server "I am sending JSON data."
    # JSON is a structured text format both Python and the server understand.
    headers = {"Content-Type": "application/json"}

    if provider == "azure":
        # AZURE AUTH: Header name is literally "api-key".
        # The value is the raw API key, no prefix.
        headers["api-key"] = api_key
        logger.info("Using Azure auth: api-key header")
    else:
        # OPENAI AUTH: Header name is "Authorization".
        # Value starts with "Bearer " (note the space) then the key.
        # "Bearer" is from RFC 6750, an internet standard for API tokens.
        headers["Authorization"] = "Bearer " + api_key
        logger.info("Using OpenAI auth: Bearer header")

    return headers


# ===========================================================================
# FUNCTION 4: call_llm_api  (THE MAIN FUNCTION -- THIS IS WHAT YOU CALL)
# ===========================================================================
# WHAT IT DOES:
#   This is the function your pipeline calls to get an AI answer.
#   It handles EVERYTHING:
#     1. Builds correct URL (prevents 404 from doubling)
#     2. Builds correct auth headers (prevents 401)
#     3. Sends request with 60-second timeout (prevents hanging)
#     4. Parses response if successful
#     5. Gives detailed troubleshooting if something goes wrong
#     6. Measures round-trip time (latency)
#
# HOW THE AI CONVERSATION FORMAT WORKS:
#   The API expects "messages" -- a list of dictionaries:
#     [
#       {"role": "system", "content": "You are a helpful assistant..."},
#       {"role": "user", "content": "What does this document say about...?"}
#     ]
#   "system" = instructions for the AI's behavior
#   "user"   = the actual question from the human
#
# INPUT:
#   endpoint_url -- Stored API URL (from credential manager)
#   api_key      -- Secret API key (from credential manager)
#   messages     -- The conversation to send (list of dicts)
#   max_tokens   -- Max response length (1 token ~ 0.75 words, default 512)
#   temperature  -- Creativity: 0.0=factual, 1.0=creative (default 0.2 for RAG)
#
# OUTPUT:
#   Dictionary with:
#     "answer"   -- AI response text (None if failed)
#     "provider" -- "azure" or "openai"
#     "model"    -- Model name from server
#     "usage"    -- Token counts (for billing tracking)
#     "latency"  -- Seconds the call took
#     "error"    -- Error message (None if success)
#
# USAGE:
#   result = call_llm_api(endpoint, key, messages)
#   if result["error"] is None:
#       print("Answer:", result["answer"])
#   else:
#       print("Problem:", result["error"])
# ---------------------------------------------------------------------------

def call_llm_api(endpoint_url, api_key, messages, max_tokens=512, temperature=0.2):
    """Make an LLM API call with auto provider detection and error handling."""

    # Start the stopwatch
    start_time = time.time()

    # Step 1: Detect provider (azure or openai)
    provider = detect_provider(endpoint_url)

    # Step 2: Build the correct, complete URL (prevents doubling)
    full_url = build_api_url(endpoint_url)

    # Step 3: Build correct auth headers (prevents 401)
    headers = build_headers(endpoint_url, api_key)

    # Step 4: Build the request body (the actual data we send)
    body = {
        "messages": messages,       # The conversation + question
        "max_tokens": max_tokens,   # Cap on response length
        "temperature": temperature, # Creativity dial (low for RAG)
    }

    # Log what we're doing, but MASK the API key for security.
    # NEVER log a full API key. Show only first 4 + last 4 characters.
    if len(api_key) > 8:
        key_preview = api_key[:4] + "..." + api_key[-4:]
    else:
        key_preview = "***"
    logger.info("API call: provider=%s url=%s key=%s", provider, full_url, key_preview)

    # Step 5: SEND THE REQUEST
    # try/except means "try this, and if something goes wrong, don't crash --
    # jump to the except block and handle the error gracefully."
    try:
        # requests.post() sends an HTTP POST to the server.
        # Think of it as mailing the letter:
        #   full_url = the address
        #   headers  = info on the envelope
        #   json=body = the letter inside (auto-converted to JSON)
        #   timeout  = give up after 60 seconds
        #   verify   = check SSL certificate (needs pip-system-certs on corp net)
        response = requests.post(
            full_url,
            headers=headers,
            json=body,
            timeout=60,
            verify=True,
        )

        # Stop the stopwatch
        latency = time.time() - start_time

        # Step 6: INTERPRET THE RESPONSE
        # HTTP status codes (standardized numbers):
        #   200 = Success
        #   401 = Unauthorized (wrong badge)
        #   404 = Not Found (wrong address)
        #   429 = Rate Limited (slow down)
        #   500 = Server Error (their problem)

        if response.status_code == 200:
            # ===== SUCCESS =====
            # Parse the JSON response back into a Python dictionary.
            data = response.json()

            # The AI's answer is nested at:
            #   data["choices"][0]["message"]["content"]
            # "choices" is a list (usually one item), [0] gets the first,
            # ["message"]["content"] is the actual text.
            return {
                "answer": data["choices"][0]["message"]["content"],
                "provider": provider,
                "model": data.get("model", "unknown"),
                "usage": data.get("usage", {}),
                "latency": latency,
                "error": None,   # None = no error = success
            }

        elif response.status_code == 401:
            # ===== 401 UNAUTHORIZED (most common problem) =====
            msg = "401 Unauthorized. Provider: " + provider + ". "
            if provider == "azure":
                msg += (
                    "TROUBLESHOOTING: "
                    "(1) Run rag-debug-url -- verify header says 'api-key' not 'Bearer'. "
                    "(2) Test on DIRECT corporate LAN, not VPN. "
                    "(3) Check if API key has expired or been rotated. "
                    "(4) In Azure portal, verify 'Cognitive Services User' role."
                )
            else:
                msg += (
                    "TROUBLESHOOTING: "
                    "(1) Check API key is correct and complete. "
                    "(2) Check account has billing/credits. "
                    "(3) Check key is for the right organization."
                )
            logger.error(msg)
            return {"answer": None, "provider": provider, "model": None,
                    "usage": None, "latency": latency, "error": msg}

        elif response.status_code == 404:
            # ===== 404 NOT FOUND (URL doubling or wrong deployment) =====
            msg = (
                "404 Not Found. URL: " + full_url + " "
                "TROUBLESHOOTING: "
                "(1) Run rag-debug-url -- compare Constructed URL with Postman. "
                "(2) Check deployment name matches Azure portal. "
                "(3) Check URL is not doubled (same path twice). "
                "(4) Check api-version is correct."
            )
            logger.error(msg)
            return {"answer": None, "provider": provider, "model": None,
                    "usage": None, "latency": latency, "error": msg}

        elif response.status_code == 429:
            # ===== 429 RATE LIMITED (too many requests) =====
            msg = "429 Rate Limited. Wait 30-60 seconds and retry."
            logger.warning(msg)
            return {"answer": None, "provider": provider, "model": None,
                    "usage": None, "latency": latency, "error": msg}

        else:
            # ===== ANY OTHER ERROR =====
            msg = "HTTP " + str(response.status_code) + ": " + response.text[:500]
            logger.error(msg)
            return {"answer": None, "provider": provider, "model": None,
                    "usage": None, "latency": latency, "error": msg}

    # --- NETWORK ERRORS (can't even reach the server) ---

    except requests.exceptions.SSLError as e:
        # SSL CERTIFICATE ERROR: Corporate proxy intercepts HTTPS and
        # Python doesn't trust the proxy's certificate.
        # FIX: pip install pip-system-certs
        msg = (
            "SSL Error. Python doesn't trust the certificate. "
            "FIX: pip install pip-system-certs. "
            "Detail: " + str(e)[:200]
        )
        logger.error(msg)
        return {"answer": None, "provider": provider, "model": None,
                "usage": None, "latency": time.time() - start_time, "error": msg}

    except requests.exceptions.Timeout:
        # TIMEOUT: Server didn't respond within 60 seconds.
        msg = (
            "Timeout after 60 seconds. "
            "FIX: (1) Try URL in browser. (2) Check firewall. (3) Retry."
        )
        logger.error(msg)
        return {"answer": None, "provider": provider, "model": None,
                "usage": None, "latency": time.time() - start_time, "error": msg}

    except requests.exceptions.ConnectionError as e:
        # CONNECTION FAILED: Can't reach server at all.
        msg = (
            "Connection failed. "
            "FIX: (1) Check network. (2) Check URL spelling. "
            "(3) Check proxy. (4) ping hostname. "
            "Detail: " + str(e)[:200]
        )
        logger.error(msg)
        return {"answer": None, "provider": provider, "model": None,
                "usage": None, "latency": time.time() - start_time, "error": msg}


# ===========================================================================
# FUNCTION 5: debug_api_config  (DIAGNOSTIC / PRE-FLIGHT CHECK)
# ===========================================================================
# WHAT IT DOES:
#   Prints what URL and headers WOULD be sent, WITHOUT making a real call.
#   Like a pre-flight checklist -- verify before takeoff.
#
# WHEN TO USE:
#   Run "rag-debug-url" from PowerShell. Do this BEFORE "rag-test-api".
#
# WHAT IT CHECKS:
#   - Provider detected (azure or openai)
#   - Constructed URL (catches doubling)
#   - Auth header format (catches 401 cause)
#   - API key length (catches truncated keys)
#   - Double slashes, missing parameters
# ---------------------------------------------------------------------------

def debug_api_config(endpoint_url, api_key):
    """Print diagnostic info. Called by the rag-debug-url command."""

    provider = detect_provider(endpoint_url)
    full_url = build_api_url(endpoint_url)

    # Mask key for display (security)
    if len(api_key) > 8:
        key_preview = api_key[:4] + "..." + api_key[-4:]
    else:
        key_preview = "***"

    print("=" * 60)
    print("API CONFIGURATION DIAGNOSTIC")
    print("=" * 60)
    print("Stored endpoint:   ", endpoint_url)
    print("Detected provider: ", provider)
    print("Constructed URL:   ", full_url)
    print("Auth header name:  ", "api-key" if provider == "azure" else "Authorization")
    print("Auth key preview:  ", key_preview)
    print("=" * 60)

    # Automated problem checks
    problems = []

    # Double slash in path (URL doubling symptom)
    url_path = full_url.replace("https://", "").replace("http://", "")
    if "//" in url_path:
        problems.append("ERROR: Double slash in URL path (sign of doubling)")

    # Missing endpoint
    if "chat/completions" not in full_url:
        problems.append("ERROR: /chat/completions missing from URL")

    # Azure missing required parameter
    if provider == "azure" and "api-version" not in full_url:
        problems.append("ERROR: api-version missing (required for Azure)")

    # Key seems too short
    if len(api_key.strip()) < 10:
        problems.append("WARNING: API key seems too short (< 10 chars)")

    # Key is empty
    if not api_key.strip():
        problems.append("ERROR: API key is empty -- run rag-store-key")

    if problems:
        print("\nPROBLEMS FOUND:")
        for p in problems:
            print("  >>", p)
    else:
        print("\nNo obvious problems. If 401 persists, test on direct LAN.")
    print("=" * 60)
