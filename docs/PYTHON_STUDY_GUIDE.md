# Python Study Guide for Non-Programmers

**Audience:** Someone who has never programmed before and wants to read and
understand the HybridRAG3 codebase.

**How to use this guide:** Work through it top to bottom. Every concept
includes a plain-English explanation, a real code snippet from this project
(with file path and line number so you can see it in context), and a
"What to notice" callout that highlights the pattern.

---

## Tier 1 -- Absolute Basics

These are the building blocks that appear on almost every line of Python code.

---

### 1.1 Variables

A **variable** is a name that points to a value. Think of it as a labeled
sticky note -- the label is the name, and the note has the value written on it.

```python
# src/core/embedder.py:107
DEFAULT_MODEL = "all-MiniLM-L6-v2"
```

Here `DEFAULT_MODEL` is the variable name and `"all-MiniLM-L6-v2"` is the
value (a text string). Whenever code later says `DEFAULT_MODEL`, Python
substitutes the value `"all-MiniLM-L6-v2"`.

> **What to notice:** Python uses `=` for assignment ("put this value into this
> name"), not for math equality. `DEFAULT_MODEL = "all-MiniLM-L6-v2"` means
> "store this text under the name DEFAULT_MODEL."

---

### 1.2 Strings

A **string** is a piece of text enclosed in quotes. Python accepts single
quotes (`'hello'`), double quotes (`"hello"`), or triple quotes for multi-line
text.

```python
# src/core/boot.py:93
lines.append(f"  Overall:  {'READY' if self.success else 'FAILED'}")
```

The `f"..."` is called an **f-string** (formatted string). Anything inside
`{curly braces}` is evaluated as Python code and its result is inserted
into the text. So if `self.success` is `True`, this produces
`"  Overall:  READY"`.

> **What to notice:** The `f` before the opening quote is what makes it an
> f-string. Without the `f`, the braces would be literal characters, not
> evaluated code.

---

### 1.3 Numbers

Python has **integers** (whole numbers) and **floats** (decimal numbers).

```python
# src/core/config.py:153-154
chunk_size: int = 1200        # Characters per chunk
overlap: int = 200            # Character overlap between consecutive chunks
```

```python
# src/core/config.py:200
temperature: float = 0.1       # Low = more focused/deterministic answers
```

`1200` and `200` are integers. `0.1` is a float. The `int` and `float`
after the colon are **type hints** -- they tell humans (and editors) what
kind of value to expect. Python itself does not enforce them.

> **What to notice:** The `#` symbol starts a **comment**. Everything after
> `#` on a line is ignored by Python. Comments explain *why* the code does
> something, not *what* it does.

---

### 1.4 Booleans

A **boolean** is a value that is either `True` or `False`. Named after
mathematician George Boole.

```python
# src/core/config.py:231-232
track_enabled: bool = True
pii_sanitization: bool = False
```

Booleans are used in decisions: "Is cost tracking turned on? Yes (True) or
No (False)?"

---

### 1.5 Lists

A **list** is an ordered collection of items enclosed in square brackets.
Items are separated by commas. You can put anything in a list -- strings,
numbers, other lists.

```python
# src/core/config.py:297-300
supported_extensions: List[str] = field(default_factory=lambda: [
    ".txt", ".md", ".csv", ".json", ".xml", ".log",
    ".pdf", ".docx", ".pptx", ".xlsx", ".eml",
    ".html", ".htm",
])
```

This is a list of file extensions (strings) that the indexer knows how to
read. Ignore the `field(default_factory=lambda: ...)` syntax for now -- that
is a dataclass pattern covered in Tier 2. The important part is the list
itself: `[".txt", ".md", ".csv", ...]`.

> **What to notice:** Lists use square brackets `[ ]`. Items are separated
> by commas. You can access items by position: `supported_extensions[0]`
> returns `".txt"` (positions start at 0, not 1).

---

### 1.6 Dictionaries

A **dictionary** (dict) is a collection of key-value pairs. Think of it as
a real dictionary: you look up a word (the key) and get its definition
(the value).

```python
# src/core/llm_router.py:172-177
payload = {
    "model": self.config.ollama.model,
    "prompt": prompt,
    "stream": False,
}
```

This dict has three keys: `"model"`, `"prompt"`, and `"stream"`. Each key
maps to a value. `payload["model"]` returns whatever `self.config.ollama.model`
evaluates to (e.g., `"phi4-mini"`).

> **What to notice:** Dicts use curly braces `{ }` with colons between keys
> and values. Keys are usually strings. This particular dict is being sent
> as a JSON request body to the Ollama server.

---

### 1.7 If / Else (Conditional Logic)

**Conditional statements** let code make decisions.

```python
# src/core/boot.py:328-334
result.success = result.online_available or result.offline_available

if not result.success:
    result.errors.append(
        "Neither online nor offline mode is available. "
        "Run 'rag-status' for diagnostics."
    )
```

Read this as plain English: "If the boot did NOT succeed, add an error
message to the list." The `not` keyword flips True to False and vice versa.

The `or` keyword means "at least one of these must be True." So the system
is considered successful if *either* online mode or offline mode is available.

> **What to notice:** The code under `if` is **indented** (shifted right by
> 4 spaces). Indentation is how Python knows which lines belong to the `if`
> block. This is different from most other languages, which use curly braces.

---

### 1.8 For Loops

A **for loop** repeats code once for each item in a collection.

```python
# src/core/boot.py:100-101
for w in self.warnings:
    lines.append(f"    [!] {w}")
```

This says: "For each warning `w` in the list `self.warnings`, append a
formatted string to `lines`." If there are 3 warnings, the indented line
runs 3 times -- once with `w` set to each warning.

> **What to notice:** `w` is a temporary variable that changes on each
> pass through the loop. You can name it anything: `w`, `warning`,
> `item` -- whatever makes the code readable.

---

### 1.9 Functions

A **function** is a reusable block of code with a name. You "define" it
once and "call" it whenever you need it.

```python
# src/core/llm_router.py:53-59
def _openai_sdk_available():
    """Check if the openai SDK is installed (lazy, no import at module load)."""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False
```

- `def` starts a function definition.
- `_openai_sdk_available` is the function name.
- `()` means it takes no parameters (no inputs).
- The triple-quoted string `"""..."""` is a **docstring** -- a built-in
  description of what the function does.
- `return True` sends the value `True` back to whoever called the function.

To use this function elsewhere, you write:
```python
if _openai_sdk_available():
    print("SDK is installed")
```

> **What to notice:** The leading underscore `_` in `_openai_sdk_available`
> is a Python convention meaning "this is private -- intended for internal
> use only, not for other files to call directly."

---

### 1.10 Parameters and Arguments

Functions can accept inputs called **parameters**.

```python
# src/core/boot.py:113
def load_config(config_path=None) -> dict:
```

- `config_path` is a parameter -- a placeholder for a value that will be
  provided when the function is called.
- `=None` means "if nobody provides a value, use `None`" (None is Python's
  way of saying "nothing" or "not set").
- `-> dict` is a type hint meaning "this function returns a dictionary."

When you call the function:
```python
config = load_config("config/my_config.yaml")  # config_path = "config/my_config.yaml"
config = load_config()                          # config_path = None (uses default)
```

The value you pass in is called an **argument**.

> **What to notice:** Parameters with defaults (like `config_path=None`) are
> **optional** -- callers can skip them. Parameters without defaults are
> **required**.

---

### 1.11 Imports

**Imports** let you use code from other files or libraries. Think of it as
borrowing a tool from someone else's toolbox.

```python
# src/core/boot.py:48-57
from __future__ import annotations

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
```

There are two styles:

1. **`import logging`** -- imports the entire `logging` module. You use it
   as `logging.getLogger(...)`.

2. **`from pathlib import Path`** -- imports just the `Path` class from the
   `pathlib` module. You use it directly as `Path(...)` instead of
   `pathlib.Path(...)`.

> **What to notice:** The boot.py file imports from three sources:
> - Python's standard library (`logging`, `os`, `sys`, `threading`, etc.)
> - The project's own files (seen later as `from src.core.config import ...`)
> - Third-party packages (`yaml`, `httpx`, `numpy`, etc.)

---

## Tier 2 -- Intermediate Python

These patterns show up throughout the project. Understanding them lets you
read any file in the codebase.

---

### 2.1 Classes

A **class** is a blueprint for creating objects. An object bundles data
(attributes) and behavior (methods) together.

```python
# src/core/llm_router.py:96-120
class OllamaRouter:
    """Route queries to local Ollama server (offline mode)."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_app_logger("ollama_router")
        self.base_url = config.ollama.base_url.rstrip("/")
        self._client = httpx.Client()
        self._health_cache = None
        self._health_ttl = 30
```

- `class OllamaRouter:` defines a new type of object called `OllamaRouter`.
- `__init__` is the **constructor** -- it runs automatically when you create
  a new instance: `router = OllamaRouter(config)`.
- `self` is a reference to the specific instance being created. Think of it
  as "me" -- `self.config` means "my config", `self.base_url` means "my URL".
- `self._client = httpx.Client()` stores an HTTP client on this instance.

After creating the object, you call its methods with the dot syntax:
```python
router = OllamaRouter(config)
available = router.is_available()    # calls the is_available() method
response = router.query("What is X?")  # calls the query() method
```

> **What to notice:** Methods defined inside a class always take `self` as
> their first parameter. When you call `router.query("What is X?")`, Python
> automatically passes the `router` object as `self`.

---

### 2.2 Inheritance

A class can **inherit** from another class, gaining all its attributes and
methods. The child class extends or customizes the parent.

```python
# src/gui/panels/query_panel.py:22-30
class QueryPanel(tk.LabelFrame):
    """Query input and answer display panel."""

    def __init__(self, parent, config, query_engine=None):
        t = current_theme()
        super().__init__(parent, text="Query Panel", padx=8, pady=8,
                         bg=t["panel_bg"], fg=t["accent"],
                         font=FONT_BOLD)
```

- `QueryPanel(tk.LabelFrame)` means QueryPanel inherits from `tk.LabelFrame`
  (a tkinter widget that draws a labeled box).
- `super().__init__(...)` calls the parent's constructor, so `QueryPanel`
  gets all the built-in LabelFrame behavior (drawing a border, displaying
  a title) for free.
- Then `QueryPanel` adds its own attributes (`self.config`, `self.query_engine`)
  and methods (like `_build_widgets`, `_on_ask`).

> **What to notice:** `super()` is how a child class calls its parent's
> methods. Without it, you would lose the parent's initialization and the
> widget would not draw properly.

---

### 2.3 Dataclasses

A **dataclass** is a shortcut for creating a class that mainly holds data.
Python automatically generates the `__init__`, `__repr__`, and other boilerplate.

```python
# src/core/llm_router.py:74-81
@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str              # The actual AI-generated answer
    tokens_in: int         # How many tokens were in the prompt
    tokens_out: int        # How many tokens the AI generated
    model: str             # Which model answered (e.g., "phi4-mini")
    latency_ms: float      # How long the call took in milliseconds
```

Without `@dataclass`, you would need to write:
```python
class LLMResponse:
    def __init__(self, text, tokens_in, tokens_out, model, latency_ms):
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.model = model
        self.latency_ms = latency_ms
```

The `@dataclass` decorator generates all of that automatically from the
field definitions.

> **What to notice:** `@dataclass` is a **decorator** -- a line starting
> with `@` placed above a class (or function) that modifies its behavior.
> Decorators are covered in section 2.8.

---

### 2.4 Dataclasses with Defaults and Factory Functions

Some dataclass fields have default values. When a default is a mutable object
(like a list or dict), you must use `field(default_factory=...)` to avoid a
Python gotcha where all instances would share the same list.

```python
# src/core/boot.py:62-85
@dataclass
class BootResult:
    boot_timestamp: str = ""
    success: bool = False
    online_available: bool = False
    offline_available: bool = False
    api_client: Optional[Any] = None
    config: Dict[str, Any] = field(default_factory=dict)
    credentials: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
```

- `success: bool = False` -- simple default. Every new BootResult starts
  with `success` set to `False`.
- `warnings: List[str] = field(default_factory=list)` -- each new BootResult
  gets its own fresh empty list. Without `default_factory`, every BootResult
  would share the *same* list object, and appending to one would affect all.

> **What to notice:** `Optional[Any]` means "this can be `None` or any type."
> `List[str]` means "a list of strings." These are type hints from the
> `typing` module.

---

### 2.5 Try / Except (Error Handling)

**Try/except** lets you handle errors gracefully instead of crashing.

```python
# src/core/boot.py:180-187
try:
    config = load_config(config_path)
    result.config = config
    logger.info("BOOT Step 1: Configuration loaded")
except Exception as e:
    result.errors.append(f"Config load failed: {e}")
    logger.error("BOOT Step 1 FAILED: %s", e)
    return result
```

Read this as: "Try to load the config. If anything goes wrong (an Exception),
catch the error, record it, log it, and return early instead of crashing."

- `try:` -- "attempt this code."
- `except Exception as e:` -- "if any error happens, catch it and call it `e`."
- `as e` stores the error object so you can inspect its message with `str(e)`.

> **What to notice:** The boot pipeline is designed to **never crash**. Every
> step wraps risky operations in try/except. This is a deliberate design
> decision documented at `src/core/boot.py:166` -- the function "catches
> all exceptions and records them in BootResult.errors so the caller can
> decide what to do."

---

### 2.6 File I/O

**File I/O** (Input/Output) means reading from or writing to files on disk.

```python
# src/core/boot.py:145-146
with open(path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
```

- `open(path, "r", encoding="utf-8")` -- opens a file for reading (`"r"`)
  using UTF-8 text encoding.
- `with ... as f:` -- the **context manager** pattern. It guarantees the
  file is properly closed when the block ends, even if an error occurs.
  Without `with`, you risk leaving files open and locked.
- `f` is the file object. `yaml.safe_load(f)` reads the file's contents.
- `or {}` means "if the result is None (empty file), use an empty dict."

Writing to a file looks similar:

```python
# src/api/routes.py:302-303
with open(config_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

The `"w"` means "write mode" (creates or overwrites the file).

> **What to notice:** Always use `with open(...)` instead of just `open(...)`.
> The `with` pattern is safer because it handles cleanup automatically.

---

### 2.7 List Comprehensions

A **list comprehension** builds a new list by transforming each item in an
existing collection, all in one concise line.

```python
# src/core/llm_router.py:895
models = [m.get("id", "") for m in data["data"] if m.get("id")]
```

This reads: "For each item `m` in `data["data"]`, if `m` has a non-empty
`"id"` key, extract that `"id"` and put it in the list."

The longer equivalent:
```python
models = []
for m in data["data"]:
    if m.get("id"):
        models.append(m.get("id", ""))
```

> **What to notice:** The structure is `[expression for item in collection if condition]`.
> The `if` part is optional -- without it, every item gets included.

---

### 2.8 Decorators

A **decorator** is a function that wraps another function (or class) to add
behavior. It is placed on the line above, starting with `@`.

```python
# src/api/routes.py:69-72
@router.get("/health", response_model=HealthResponse)
async def health():
    """Fast health check. Returns 200 if the server is running."""
    return HealthResponse(status="ok", version=_version())
```

The `@router.get("/health", ...)` decorator tells FastAPI: "When someone
sends a GET request to `/health`, call this function and return its result
as an HTTP response." Without the decorator, this would just be an ordinary
function that nobody calls.

You have already seen another decorator:
```python
@dataclass
class LLMResponse:
    ...
```

Here `@dataclass` transforms a plain class into one with auto-generated
`__init__`, `__repr__`, etc.

> **What to notice:** Decorators are used heavily in two places in this
> project: dataclasses (`@dataclass`) and FastAPI routes (`@router.get`,
> `@router.post`, `@router.put`). They save you from writing boilerplate code.

---

### 2.9 Type Hints

**Type hints** are annotations that describe what type a variable, parameter,
or return value should be. Python does not enforce them at runtime -- they
are documentation for humans and code editors.

```python
# src/core/config.py:533-534
def validate_config(config: Config) -> List[str]:
    """Check a Config object for problems. Returns a list of error messages."""
    errors: List[str] = []
```

- `config: Config` -- the parameter `config` should be a `Config` object.
- `-> List[str]` -- the function returns a list of strings.
- `errors: List[str] = []` -- the local variable `errors` is a list of
  strings, starting empty.

Common type hints in this project:

| Hint | Meaning |
|------|---------|
| `str` | text string |
| `int` | whole number |
| `float` | decimal number |
| `bool` | True or False |
| `List[str]` | list of strings |
| `Dict[str, Any]` | dictionary with string keys and any-type values |
| `Optional[str]` | either a string or None |
| `Tuple[bool, float]` | a tuple of a bool and a float |

> **What to notice:** Type hints make code self-documenting. When you see
> `def query(self, prompt: str) -> Optional[LLMResponse]`, you immediately
> know: it takes a string and returns either an LLMResponse or None.

---

### 2.10 Properties

A **property** is a method that looks like an attribute. You access it
without parentheses, but behind the scenes it runs code.

```python
# src/core/config.py:369-371
@property
def hallucination_guard_enabled(self) -> bool:
    return self.hallucination_guard.enabled
```

You use this as `config.hallucination_guard_enabled` (no parentheses),
but it actually calls the method, which reads `self.hallucination_guard.enabled`
and returns its value.

> **What to notice:** Properties are useful for providing simpler access to
> deeply nested values. Instead of writing
> `config.hallucination_guard.enabled` everywhere, callers can write
> `config.hallucination_guard_enabled`.

---

### 2.11 Enums

An **enum** (enumeration) is a set of named constants. It prevents bugs
where you accidentally use an invalid string like `"ofline"` (typo) instead
of `"offline"`.

```python
# src/core/network_gate.py:69-81
class NetworkMode(Enum):
    """
    The three allowed network access levels.
    """
    OFFLINE = "offline"     # Localhost only (Ollama)
    ONLINE = "online"       # Localhost + configured API endpoint
    ADMIN = "admin"         # Unrestricted (maintenance only)
```

Instead of comparing strings (`if mode == "offline"`) you compare enum values
(`if self._mode == NetworkMode.OFFLINE`). If you typo the name, Python
crashes immediately with `AttributeError: NetworkMode has no member 'OFLINE'`
instead of silently doing the wrong thing.

> **What to notice:** Enums are used in the network gate to make sure only
> the three recognized modes are ever used. Each member has a `.value`
> attribute: `NetworkMode.OFFLINE.value` returns `"offline"`.

---

### 2.12 Threading

**Threading** lets you run multiple pieces of code at the same time. This
is useful when one task is slow (like loading a model) and you want the
user to see something (like a window) while it runs.

```python
# src/gui/launch_gui.py:288-294
backend_thread = threading.Thread(
    target=_load_backends, args=(app, logger), daemon=True,
)
backend_thread.start()

# -- Step 5: Run the GUI event loop --
app.mainloop()
```

- `threading.Thread(target=..., args=...)` creates a new thread that will
  run the function `_load_backends` with arguments `(app, logger)`.
- `daemon=True` means "kill this thread when the main program exits"
  (so it does not keep running in the background forever).
- `.start()` begins execution. The main program continues immediately to
  `app.mainloop()` without waiting for backends to finish.

This is why the GUI window appears instantly -- it does not wait for the
16-second model load.

> **What to notice:** The boot pipeline uses a similar pattern at
> `src/core/boot.py:317-319` to check Ollama without blocking startup.
> Threading is used throughout the project to keep the user interface
> responsive.

---

### 2.13 Logging

**Logging** records events as the program runs. Unlike `print()`, log
messages include timestamps, severity levels, and can be written to files
for later review.

```python
# src/core/boot.py:59
logger = logging.getLogger(__name__)
```

```python
# src/core/boot.py:179-186
logger.info("BOOT Step 1: Loading configuration...")
try:
    config = load_config(config_path)
    result.config = config
    logger.info("BOOT Step 1: Configuration loaded")
except Exception as e:
    result.errors.append(f"Config load failed: {e}")
    logger.error("BOOT Step 1 FAILED: %s", e)
```

- `logging.getLogger(__name__)` creates a logger named after the current file.
- `logger.info(...)` records an informational message.
- `logger.error(...)` records an error (more serious than info).
- `logger.warning(...)` records a warning (between info and error).

Severity levels from lowest to highest: `DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`.

> **What to notice:** `__name__` is a special Python variable that holds
> the current module's name. For `src/core/boot.py`, it would be
> `"src.core.boot"`. This helps you see which file generated each log line.

---

### 2.14 The `__post_init__` Method

In a dataclass, `__post_init__` runs automatically right after the
auto-generated `__init__` finishes. It lets you add custom logic.

```python
# src/core/config.py:82-105
def __post_init__(self) -> None:
    # If HYBRIDRAG_DATA_DIR is set, use it to build database + cache paths
    data_dir = os.getenv("HYBRIDRAG_DATA_DIR")
    if data_dir:
        if not self.database:
            self.database = os.path.join(data_dir, "hybridrag.sqlite3")
        if not self.embeddings_cache:
            self.embeddings_cache = data_dir

    # Clean up paths: expand %VARIABLES% and normalize slashes
    if self.database:
        self.database = os.path.normpath(os.path.expandvars(self.database))
```

The `@dataclass` decorator creates `__init__` for you (it sets
`self.database`, `self.embeddings_cache`, `self.source_folder` from
constructor arguments). Then `__post_init__` runs and says: "OK, now let
me also check for environment variable overrides and clean up the paths."

> **What to notice:** This pattern lets you layer logic on top of auto-generated
> code. The dataclass handles the boring part (storing fields), and
> `__post_init__` handles the smart part (environment variable overrides,
> path normalization).

---

## Tier 3 -- Libraries Used in This Project

These are third-party (and standard-library) tools that show up in the
codebase. You do not need to memorize their APIs -- just understand what
role each one plays.

---

### 3.1 YAML (Configuration Files)

**YAML** is a file format for structured settings. It is more readable
than JSON for configuration because it uses indentation instead of braces.

The file `config/default_config.yaml`:
```yaml
mode: offline
ollama:
  base_url: http://localhost:11434
  model: phi4-mini
  timeout_seconds: 600
embedding:
  model_name: all-MiniLM-L6-v2
  dimension: 384
  batch_size: 16
  device: cpu
```

And the Python code that reads it:

```python
# src/core/boot.py:126-127
import yaml

# ...
with open(path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}
```

`yaml.safe_load(f)` reads the YAML file and returns a Python dictionary:
```python
{
    "mode": "offline",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "phi4-mini",
        "timeout_seconds": 600,
    },
    "embedding": {
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        ...
    },
}
```

> **What to notice:** YAML uses indentation to show nesting (like Python
> itself). The `safe_load` function is used instead of `load` because
> `load` can execute arbitrary code in a YAML file (a security risk).

---

### 3.2 pathlib.Path (File Paths)

**pathlib** is Python's modern way to work with file system paths. It
replaces messy string concatenation with clean, cross-platform operations.

```python
# src/core/boot.py:132
project_root = Path(__file__).resolve().parent.parent.parent
```

Breaking this down:
- `Path(__file__)` -- the path to the current Python file (`boot.py`).
- `.resolve()` -- converts to an absolute path (no `..` or `.`).
- `.parent` -- goes up one directory. Called three times: `boot.py` is in
  `src/core/`, so `.parent.parent.parent` goes up to the project root.

Other common Path operations:
```python
path = Path("config") / "default_config.yaml"   # Joins paths with /
path.exists()                                     # True if file exists
path.name                                        # "default_config.yaml"
path.suffix                                      # ".yaml"
```

> **What to notice:** The `/` operator joins path segments. This is cleaner
> and more cross-platform than `os.path.join("config", "default_config.yaml")`.

---

### 3.3 os and os.environ (Operating System Interface)

The **os** module lets Python interact with the operating system --
environment variables, file paths, directory creation.

```python
# src/core/embedder.py:86-89
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
```

- `os.environ` is a dictionary of all environment variables.
- `setdefault(key, value)` means "set this variable only if it is not already
  set." This ensures the HuggingFace model library never reaches the internet,
  while still allowing someone to override the setting if they have a reason.

Other common `os` operations in this project:
```python
os.getenv("HYBRIDRAG_DATA_DIR")      # Read an environment variable (or None)
os.path.join(dir, "file.db")         # Join path components
os.makedirs(path, exist_ok=True)     # Create directories (no error if exists)
os.path.exists(path)                 # Check if a file/folder exists
```

> **What to notice:** Environment variables are how you configure per-machine
> settings without editing code. The project uses them for data directories,
> API endpoints, and security lockdowns.

---

### 3.4 SQLite (Database)

**SQLite** is a database engine built into Python. It stores structured
data in a single file (no server required).

```python
# src/core/vector_store.py:55-56
import sqlite3

# Connection and query (simplified from the actual code):
conn = sqlite3.connect("hybridrag.sqlite3")
cursor = conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,))
row = cursor.fetchone()
```

- `sqlite3.connect(...)` opens (or creates) a database file.
- `conn.execute(sql, params)` runs a SQL query. The `?` is a placeholder
  for `chunk_id` -- this prevents SQL injection attacks.
- `cursor.fetchone()` retrieves one row of results.

> **What to notice:** The `?` placeholder is critical for security. Never
> put user input directly into SQL strings like
> `f"SELECT * FROM chunks WHERE id = '{user_input}'"` -- that allows
> SQL injection attacks.

---

### 3.5 httpx (HTTP Client)

**httpx** is a modern HTTP client library. It is used to talk to the local
Ollama server.

```python
# src/core/llm_router.py:116
self._client = httpx.Client()
```

```python
# src/core/llm_router.py:180-184
resp = self._client.post(
    f"{self.base_url}/api/generate",
    json=payload,
    timeout=self.config.ollama.timeout_seconds,
)
resp.raise_for_status()
```

- `httpx.Client()` creates a persistent HTTP client that reuses connections.
- `.post(url, json=payload, timeout=...)` sends a POST request with a JSON body.
- `.raise_for_status()` throws an error if the server returned an error
  code (like 404 or 500).
- `resp.json()` parses the response body as JSON into a Python dict.

> **What to notice:** `httpx.Client()` is stored on the class (`self._client`)
> and reused across multiple calls. This is more efficient than creating a
> new connection each time.

---

### 3.6 numpy (Numerical Arrays)

**numpy** is a library for fast math on large arrays of numbers. HybridRAG
uses it for embedding vectors.

```python
# src/core/embedder.py:156
return np.zeros((0, self.dimension), dtype=np.float32)
```

```python
# src/core/embedder.py:158-164
embeddings = self.model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
)
return embeddings.astype(np.float32, copy=False)
```

- `np.zeros((0, 384))` creates an empty array with 0 rows and 384 columns.
- `.astype(np.float32, copy=False)` converts the data type to 32-bit floats
  without making a copy if it is already float32.
- The embedding model returns an array of shape `(N, 384)` where N is the
  number of texts and 384 is the embedding dimension.

Key numpy concepts for this project:
- **Shape**: `(1000, 384)` means 1000 rows, 384 columns -- 1000 text chunks,
  each represented by 384 numbers.
- **dtype**: `float32` (4 bytes per number) vs `float16` (2 bytes) -- the
  project uses float16 on disk to save space.
- **memmap**: memory-mapped files -- numpy reads from disk on demand instead
  of loading everything into RAM.

> **What to notice:** The vector store (`src/core/vector_store.py`) uses
> numpy memmap files so 40,000 embedding vectors do not need to fit in RAM.

---

### 3.7 tkinter (GUI)

**tkinter** is Python's built-in GUI library. It creates desktop windows
with buttons, text fields, and other widgets.

```python
# src/gui/app.py:43-57
class HybridRAGApp(tk.Tk):
    """Main application window for HybridRAG v3."""

    def __init__(self, boot_result=None, config=None, ...):
        super().__init__()

        self.title("HybridRAG v3")
        self.geometry("780x720")
        self.minsize(640, 500)
```

- `tk.Tk` is the main window class. Inheriting from it gives you a window.
- `self.title(...)` sets the window title bar text.
- `self.geometry("780x720")` sets the initial window size in pixels.

Common widget creation:
```python
# src/gui/panels/query_panel.py:85-89
self.question_entry = tk.Entry(
    row2, font=FONT, bg=t["input_bg"], fg=t["input_fg"],
    insertbackground=t["fg"], relief=tk.FLAT, bd=2,
)
self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
```

- `tk.Entry(parent, ...)` creates a text input field.
- `.pack(...)` places it in the window using the "pack" layout manager.
- `fill=tk.X` means "stretch to fill available horizontal space."

> **What to notice:** The GUI is built at `src/gui/app.py` with panels split
> into separate files under `src/gui/panels/`. This keeps each file focused
> on one piece of the interface.

---

### 3.8 Pydantic (Data Validation for APIs)

**Pydantic** is a library that validates data using Python type annotations.
FastAPI uses it to automatically validate request and response data.

```python
# src/api/models.py:20-27
class QueryRequest(BaseModel):
    """POST /query request body."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about your documents.",
    )
```

- `BaseModel` is the Pydantic base class. Any class inheriting from it gets
  automatic validation.
- `Field(...)` defines constraints: the question must be between 1 and 2000
  characters.
- The `...` (ellipsis) means "this field is required" (no default value).

If someone sends a request with an empty question or one that is 3000
characters long, Pydantic automatically returns a 422 error with a clear
message. No manual validation code needed.

> **What to notice:** All request and response models live in
> `src/api/models.py`. This separates "what shape the data must be" from
> "what to do with it" (routes.py).

---

### 3.9 FastAPI (Web Server)

**FastAPI** is a web framework for building REST APIs. It uses decorators
to map URLs to Python functions.

```python
# src/api/routes.py:69-72
@router.get("/health", response_model=HealthResponse)
async def health():
    """Fast health check. Returns 200 if the server is running."""
    return HealthResponse(status="ok", version=_version())
```

```python
# src/api/routes.py:135-159
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Ask a question about your indexed documents."""
    s = _state()
    if not s.query_engine:
        raise HTTPException(status_code=503, detail="Query engine not initialized")

    result = s.query_engine.query(req.question)

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        ...
    )
```

- `@router.get("/health")` -- handle GET requests to `/health`.
- `@router.post("/query")` -- handle POST requests to `/query`.
- `async def` -- FastAPI uses Python's async syntax for efficiency, but
  the functions read like normal synchronous code.
- `response_model=QueryResponse` -- FastAPI validates the return value
  against this Pydantic model and generates API documentation automatically.
- `raise HTTPException(...)` -- returns an HTTP error to the client.

> **What to notice:** FastAPI, Pydantic, and type hints work together:
> Pydantic validates the input (`QueryRequest`), your function processes it,
> Pydantic validates the output (`QueryResponse`), and FastAPI serializes
> it to JSON. Three libraries, one smooth pipeline.

---

## Tier 4 -- Putting It Together

Now that you know the building blocks, here is how they combine into the
two most important flows in HybridRAG.

---

### 4.1 How the Boot Pipeline Works

When HybridRAG starts, it needs to check that everything is ready before
accepting queries. The boot pipeline (`src/core/boot.py`) does this in 5 steps.

**Step 1: Load configuration**
```
boot_hybridrag()
  -> load_config()
     -> Finds config/default_config.yaml
     -> yaml.safe_load() reads it into a Python dict
     -> Returns the dict
```
This uses: YAML (3.1), pathlib (3.2), file I/O (2.6), try/except (2.5).

**Step 2: Resolve credentials**
```
  -> resolve_credentials(config)
     -> Checks Windows Credential Manager (keyring)
     -> Checks environment variables (AZURE_OPENAI_API_KEY, etc.)
     -> Checks config dict
     -> Returns ApiCredentials dataclass
```
This uses: dataclasses (2.3), os.environ (3.3), try/except (2.5).

**Step 2.5: Configure the network gate**
```
  -> configure_gate(mode, api_endpoint, allowed_prefixes)
     -> Gets the singleton NetworkGate
     -> Sets mode to OFFLINE, ONLINE, or ADMIN
     -> Builds the allowed-hosts list from the API endpoint
```
This uses: enums (2.11), singleton pattern, classes (2.1).

**Step 3: Build API client** (if credentials are available)
```
  -> ApiClientFactory(config).build(credentials)
     -> Detects Azure vs standard OpenAI
     -> Creates the appropriate SDK client
     -> Returns a client object (or records an error)
```

**Step 4: Check Ollama** (non-blocking)
```
  -> threading.Thread(target=_check_ollama)
     -> Sends GET to http://localhost:11434/api/tags
     -> If response is 200, marks offline mode as available
     -> Thread is joined with 0.5s timeout
```
This uses: threading (2.12), try/except (2.5), httpx-like access via urllib.

**Step 5: Return BootResult**
```
  -> result.success = online_available or offline_available
  -> Return BootResult dataclass with all status info
```

The key design principle: **the boot pipeline never crashes**. Every step is
wrapped in try/except. If credentials are missing, online mode is simply
marked "unavailable" but offline mode still works. The caller gets a
`BootResult` object with all the details and decides what to do.

---

### 4.2 How a Query Flows End-to-End

When you type a question in the GUI and click "Ask", here is everything
that happens:

```
User types: "What is the operating frequency?"
                |
                v
[QueryPanel] (src/gui/panels/query_panel.py)
  _on_ask() is triggered by Enter key or Ask button
  Spawns a background thread so the GUI doesn't freeze
                |
                v
[QueryEngine] (src/core/query_engine.py:111)
  query("What is the operating frequency?")
                |
     +----------+----------+
     |                     |
     v                     |
  Step 1: SEARCH           |
  retriever.search(query)  |
     |                     |
     v                     |
  [Retriever] (src/core/retriever.py)
    1. embed_query() -- convert question to 384-number vector
    2. Vector search -- find chunks with similar vectors
    3. BM25 search  -- find chunks with matching keywords
    4. RRF fusion   -- merge both ranked lists
    5. Return top-k SearchHit objects
                |
                v
  Step 2: BUILD CONTEXT
  retriever.build_context(search_results)
    Combines the top chunk texts into one block of text
                |
                v
  Step 3: BUILD PROMPT
  _build_prompt(user_query, context)
    Wraps the context + question in the 9-rule prompt template
    "You are a precise technical assistant. Answer using ONLY..."
                |
                v
  Step 4: LLM CALL
  llm_router.query(prompt)
     |
     +-- OFFLINE MODE:  OllamaRouter.query()
     |     httpx POST to localhost:11434/api/generate
     |     Returns LLMResponse(text=..., tokens_in=..., ...)
     |
     +-- ONLINE MODE:   APIRouter.query()
           openai SDK chat.completions.create()
           Returns LLMResponse(text=..., tokens_in=..., ...)
                |
                v
  Step 5: CALCULATE COST
  _calculate_cost(llm_response)
    Offline: $0.00
    Online: (tokens_in / 1000) * rate + (tokens_out / 1000) * rate
                |
                v
  Step 6: LOG + RETURN
  Return QueryResult(
      answer="The operating frequency is 10 MHz...",
      sources=[{"path": "System_Spec.pdf", "chunks": 3, ...}],
      chunks_used=5,
      latency_ms=1234.5,
      cost_usd=0.003,
      mode="offline",
  )
                |
                v
[QueryPanel] displays the answer in the text area
```

**Summary of patterns used in the query flow:**

| Pattern | Where |
|---------|-------|
| Classes (2.1) | QueryEngine, Retriever, OllamaRouter, LLMRouter |
| Dataclasses (2.3) | QueryResult, LLMResponse, SearchHit |
| Try/except (2.5) | Every step has error handling |
| Threading (2.12) | GUI runs query in background thread |
| Type hints (2.9) | `def query(self, user_query: str) -> QueryResult` |
| F-strings (1.2) | `f"{self.base_url}/api/generate"` |
| Dictionaries (1.6) | Request payload, source metadata |
| Conditionals (1.7) | `if mode == "online"` to pick the right backend |
| httpx (3.5) | Ollama HTTP calls |
| numpy (3.6) | Embedding vector operations |
| SQLite (3.4) | Chunk text retrieval |

---

### 4.3 Reading a New File in the Project

When you open any file in this codebase, here is a systematic way to
read it:

1. **Read the file header comment.** Every file starts with a block comment
   explaining what it does, why it exists, and what design decisions were made.

2. **Look at the imports.** They tell you what tools the file uses and which
   other project files it depends on.

3. **Find the classes.** Most files define one or two classes. Read their
   docstrings to understand their purpose.

4. **Find the main entry points.** Look for:
   - The `__init__` method (constructor) to see how the class is set up.
   - Public methods (no underscore prefix) to see what the class does.
   - Private methods (underscore prefix) for internal implementation details.

5. **Follow the data.** Start from a public method and trace what happens:
   - What parameters come in?
   - What gets called?
   - What gets returned?

6. **Look for patterns from this guide.** Once you recognize dataclasses,
   try/except blocks, threading, and decorator-based routes, you can read
   any file in the project.

---

## Quick Reference: Python Glossary

| Term | Meaning |
|------|---------|
| **argument** | A value passed to a function when you call it |
| **attribute** | A variable that belongs to an object (`self.config`) |
| **boolean** | `True` or `False` |
| **class** | A blueprint for creating objects |
| **constructor** | The `__init__` method that runs when you create an object |
| **dataclass** | A class that auto-generates `__init__` from field definitions |
| **decorator** | A `@something` line that modifies a function or class |
| **dict** | A collection of key-value pairs (`{"name": "value"}`) |
| **docstring** | A `"""triple-quoted"""` description inside a function/class |
| **enum** | A fixed set of named constants |
| **f-string** | A string with `f"..."` that evaluates `{expressions}` |
| **float** | A decimal number (`3.14`) |
| **function** | A reusable block of code defined with `def` |
| **import** | A statement that makes code from another file available |
| **int** | A whole number (`42`) |
| **list** | An ordered collection (`[1, 2, 3]`) |
| **method** | A function that belongs to a class |
| **module** | A Python file (`.py`) that can be imported |
| **None** | Python's "nothing" value |
| **parameter** | A variable in a function definition that receives a value |
| **property** | A method disguised as an attribute (using `@property`) |
| **self** | A reference to the current object instance |
| **string** | A piece of text (`"hello"`) |
| **tuple** | An immutable ordered collection (`(1, 2, 3)`) |
| **type hint** | An annotation describing expected types (`x: int = 5`) |
| **variable** | A name that stores a value |
