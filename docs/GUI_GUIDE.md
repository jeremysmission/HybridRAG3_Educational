# HybridRAG v3 -- GUI User Guide (Dark-Mode Prototype)

Last Updated: 2026-02-21

This guide covers every panel, button, slider, and toggle in the HybridRAG
GUI. It is written for someone who has never seen the application before.

The GUI ships with a **dark theme** (default) and a **light theme**, toggled
with one click.

---

## Table of Contents

1. [Launching the GUI](#1-launching-the-gui)
2. [Window Layout Overview](#2-window-layout-overview)
3. [Title Bar, Mode Toggle, and Theme Switch](#3-title-bar-mode-toggle-and-theme-switch)
4. [Query Panel](#4-query-panel)
5. [Index Panel](#5-index-panel)
6. [Status Bar](#6-status-bar)
7. [Admin Menu](#7-engineering-menu)
8. [Menu Bar](#8-menu-bar)
9. [Dark Mode and Light Mode](#9-dark-mode-and-light-mode)
10. [How to Run Your First Query](#10-how-to-run-your-first-query)
11. [How to Index a New Folder](#11-how-to-index-a-new-folder)
12. [Online vs Offline Mode](#12-online-vs-offline-mode)
13. [Keyboard Shortcuts](#13-keyboard-shortcuts)
14. [Common Problems and What They Mean](#14-common-problems-and-what-they-mean)

---

## Before You Start: Opening PowerShell

If you have never used PowerShell before, follow these steps exactly:

1. Press **Win+X** on your keyboard (hold the Windows key, tap X).
   A menu appears in the bottom-left corner.
2. Click **Terminal** (or **Windows PowerShell**).
   A dark window with a blinking cursor appears. This is PowerShell.
3. Type the following and press **Enter**:
   ```
   cd "{PROJECT_ROOT}"
   ```
   This tells PowerShell to go to the project folder.
4. Type the following and press **Enter**:
   ```
   . .\start_hybridrag.ps1
   ```
   Note the **dot-space-dot** at the start. This loads the HybridRAG
   environment (paths, security, command aliases). You will see several
   `[OK]` messages scroll past.
5. You are now ready to launch the GUI (Section 1 below) or run any
   `rag-` command.

**Tip:** After step 4, the PowerShell window stays open in the
background. Do not close it -- the GUI needs it running.

---

## 1. Launching the GUI

From the PowerShell terminal you opened above, run:

```
python src/gui/launch_gui.py
```

Or from PowerShell:

```
.\tools\launch_gui.ps1
```

What happens when you launch:

1. The boot pipeline runs (loads config, checks credentials, configures the
   network gate). This takes 2-4 seconds.
2. The GUI window opens immediately.
3. Heavy backends (embedding model, vector store, query engine) load in a
   background thread. While they load, you can look around the GUI and toggle
   modes, but queries and indexing are not available yet.
4. Once backends finish loading, the query panel and index panel become
   functional. The status bar updates to reflect the live system state.

On an 8 GB RAM laptop, the background loading can take 30-60 seconds because
the embedding model (all-MiniLM-L6-v2, about 87 MB) and PyTorch must load
into memory. On a 64 GB workstation this is nearly instant.

---

## 2. Window Layout Overview

The window is 780 x 720 pixels by default (minimum 640 x 500, resizable).
It has five regions stacked top to bottom:

```
+--------------------------------------------------------------+
| Menu bar:  File | Engineering | Help                         |
+--------------------------------------------------------------+
| TITLE BAR                                                    |
| "HybridRAG v3"  Mode: [OFFLINE] [ONLINE]   Theme: [Light]   |
+--------------------------------------------------------------+
| QUERY PANEL                                                  |
| Use case dropdown, model display, question box,              |
| answer area, sources, latency metrics                        |
+--------------------------------------------------------------+
| INDEX PANEL                                                  |
| Folder picker, Start/Stop, progress bar, last run info       |
+--------------------------------------------------------------+
| STATUS BAR                                                   |
| LLM: phi4-mini (Ollama) | Ollama: Ready | [*] Gate: OFFLINE |
+--------------------------------------------------------------+
```

The default appearance is **dark mode**: dark gray backgrounds (#1e1e1e),
white text, and blue accent buttons (#0078d4). Click the Theme button in
the title bar to switch to light mode.

---

## 3. Title Bar, Mode Toggle, and Theme Switch

At the top of the window you see, left to right:

- **"HybridRAG v3"** -- the application name (13pt bold).
- **Mode:** with two toggle buttons: **OFFLINE** and **ONLINE**.
- **Theme:** with a toggle button showing **"Light"** or **"Dark"** (right side).

The active mode button is green (#4caf50 dark theme, #388e3c light theme).
The inactive mode button is gray.

### OFFLINE button

Click this to switch to offline mode. This is always safe and requires no
confirmation. In offline mode:

- Queries go to Ollama running on your local machine (localhost:11434).
- No data leaves your computer.
- The network gate blocks all external connections.
- The embedding model and vector search run locally.

### ONLINE button

Click this to switch to online mode. Before the switch happens, the system
checks whether you have stored an API key and an API endpoint in Windows
Credential Manager. If either is missing, you get a dialog box that says:

```
Cannot switch to online mode.

Missing: API key, API endpoint

Run rag-store-key and rag-store-endpoint from PowerShell first,
then try again.
```

If credentials are present, the switch succeeds. In online mode:

- Queries go to your configured API endpoint (Azure OpenAI, OpenRouter, etc.).
- Only the question text and the top retrieved document chunks are sent.
- Full documents, file paths, and the database are never sent.
- The network gate opens only for the configured endpoint. All other
  external connections remain blocked.

Switching back to offline is always one click with no confirmation needed.

### Theme toggle button

On the right side of the title bar, a button labeled **"Light"** (when in
dark mode) or **"Dark"** (when in light mode). Click to switch themes.

- **Dark mode** (default): Dark gray background (#1e1e1e), white text,
  blue accent buttons. Easier on the eyes in low-light environments.
- **Light mode**: White/light gray background (#f0f0f0), black text, same
  blue accent buttons. Better for brightly lit offices or projector demos.

The theme switches instantly. All panels, menus, the status bar, and the
engineering menu (if open) update in place without rebuilding the window.

See [Section 9: Dark Mode and Light Mode](#9-dark-mode-and-light-mode) for
the full color reference.

---

## 4. Query Panel

This is the largest panel in the window and the one you use most. It has
seven parts from top to bottom.

### Use case dropdown

A dropdown list with nine options:

| Label                  | Key   | What it means                                         |
|------------------------|-------|-------------------------------------------------------|
| Software Engineering   | sw    | Code, debugging, algorithms. 90% engineering weight.  |
| Engineering / STEM     | eng   | Math, technical analysis, RAG queries. 80% eng.       |
| Systems Administration | sys   | Scripts, configs, security, networking. 70% eng.      |
| Drafting / AutoCAD     | draft | Technical specs, structured output. 75% eng.          |
| Logistics Analyst      | log   | Data analysis, optimization, spreadsheets. 60% eng.   |
| Program Management     | pm    | Documentation, scheduling, reporting. 75% general.    |
| Field Engineer         | fe    | Site surveys, equipment install, safety. 72.5% eng.   |
| Cybersecurity Analyst  | cyber | Incident response, threat analysis, SIEM. 72.5% eng. |
| General AI             | gen   | World knowledge, creative writing, broad reasoning.   |

The default is **Engineering / STEM**.

**What it does in the background:** When you change the use case, the system
picks the best available model for that role using weighted scoring. An
engineering-heavy use case prefers models that score high on code and math
benchmarks. A general-heavy use case prefers models that score high on world
knowledge and writing. The model name updates automatically in the Model
field below.

**When to change it:** Pick the use case that matches the kind of question
you are about to ask. If you are looking up a part number from a technical
manual, use Engineering / STEM. If you are drafting a status report, use
Program Management. If you are not sure, leave it on Engineering / STEM --
it works well for most document search tasks.

### Model field (read-only)

Shows the auto-selected model name, for example:

```
gpt-4o (auto-selected)
```

You cannot type in this field. It updates automatically when you change the
use case dropdown. The "(auto-selected)" suffix tells you the system chose
this model, not you.

**What it does in the background:** Calls `select_best_model()` with the
current use case key and the list of available deployments. The function
scores each model against the use case weights and returns the best match.

### Question entry

A text box where you type your question. When you first see it, it shows
gray placeholder text: "Ask a question..."

Click the box (or tab into it) and the placeholder disappears. Type your
question in plain English. For example:

```
What is the operating frequency of the XR-7 transceiver?
```

Press **Enter** or click the **Ask** button to submit.

### Ask button

Submits your question. When you click it:

1. The button grays out (disabled) so you cannot double-submit.
2. A network indicator appears below the question box:
   - In offline mode: **"Querying local model..."** (gray text)
   - In online mode: **"Sending to API..."** (gray text)
3. The query runs in a background thread. The window stays responsive.
4. When the answer comes back, the button re-enables and the indicator
   disappears.

**The network indicator is important.** If you see "Sending to API..." it
means data is leaving your machine right now. If you are on a restricted
network and did not intend to go online, click OFFLINE immediately.

### Answer area

A scrollable text box that displays the answer. You can select text and
copy it (Ctrl+C). If the query failed, the error appears here in red with a
**[FAIL]** prefix.

The answer comes from the LLM (Ollama locally or the API remotely). The LLM
sees only the top retrieved chunks from your indexed documents and your
question. It is instructed to answer using only those chunks. If the answer
is not in the documents, it should say so.

### Sources line

Below the answer, a line shows which documents were used:

```
Sources: Manual.pdf (3 chunks), Datasheet.xlsx (2 chunks)
```

This tells you which files contributed to the answer and how many chunks from
each file were relevant. If no sources matched, it shows "Sources: (none)".

### Metrics line

Below the sources:

```
Latency: 1,234 ms | Tokens in: 450 | Tokens out: 120
```

- **Latency:** Total round-trip time in milliseconds. Offline on CPU can
  be 30,000-180,000 ms. Online is typically 3,000-15,000 ms.
- **Tokens in:** How many tokens (roughly words) were sent to the LLM.
  This includes the question plus all the retrieved chunks.
- **Tokens out:** How many tokens the LLM generated in its answer.

---

## 5. Index Panel

This panel lets you point the system at a folder of documents and build
(or update) the search index.

### Source folder field

A text box showing the path to your documents folder. Type a path directly
or use the Browse button.

### Browse button

Opens a standard folder picker dialog. Select a folder and its path
appears in the source folder field. Clicking Browse does **not** start
indexing -- it only sets the path.

### Start Indexing button

Click this to begin indexing. Before it starts, it checks:

1. Is the folder path filled in? If blank: **"[FAIL] No folder selected"**
2. Does the folder exist on disk? If not: **"[FAIL] Folder does not exist"**
3. Is the indexer initialized? If the backends are still loading:
   **"[FAIL] Indexer not initialized. Run boot first."**

If all checks pass:

1. The Start button disables and the Stop button enables.
2. The progress label shows "Starting..."
3. The system scans the folder for supported file types:
   `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`, `.csv`, `.json`,
   `.xml`, `.log`, `.eml`
4. Each file is read, split into ~1,200 character chunks with 200 character
   overlap, embedded into 384-dimension vectors, and stored in the local
   SQLite database.
5. The progress bar advances as each file completes.
6. When done, the "Last run" line updates with timestamp and chunk count.

**What happens in the background:** The indexer runs in a separate thread so
the GUI stays responsive. A progress callback updates the progress bar after
each file. Files that have not changed since the last indexing run are
skipped automatically (deterministic hash comparison).

**Network access:** NONE. Indexing is entirely local. The embedding model
runs on your CPU (or GPU if configured). No documents are sent anywhere.

### Stop button

Click this to stop indexing after the current file finishes. It does not
abort mid-file -- it waits for the file being processed to complete, then
stops. The label shows "Stopping after current file..." in orange while
it waits.

### Progress bar

A horizontal bar that fills from left to right as files are processed. Next
to it is a counter: "47 / 103 files".

### Progress label

Shows the name of the file currently being processed:

```
Processing: Engineer_Calibration_Guide.pdf
```

If a file has an error, it shows in orange:

```
[WARN] Error on corrupted_file.pdf: Failed to parse PDF
```

Errors on individual files do not stop the overall indexing run. The system
logs the error and moves to the next file.

### Last run label

After indexing completes:

```
Last run: 2026-02-21 14:30 | 4,335 chunks indexed | 287s
```

---

## 6. Status Bar

The narrow bar at the very bottom of the window. It updates automatically
every 5 seconds by polling the LLM router for current system state.

### LLM indicator

Shows which model is active and where it runs:

- `LLM: gpt-4o (azure_openai)` -- online mode, Azure backend
- `LLM: phi4-mini (Ollama)` -- offline mode, local Ollama
- `LLM: Not configured` -- no backend available
- `LLM: Not initialized` -- backends still loading

### Ollama indicator

- **Ollama: Ready** (green) -- Ollama is running on localhost and responding.
- **Ollama: Offline** (gray) -- Ollama is not running or not reachable.

If you plan to use offline mode, Ollama must be running. Start it in a
separate terminal with `ollama serve` if it is not already running.

### Gate indicator

- **Gate: ONLINE** (green, green dot) -- the network gate is open for the
  configured API endpoint.
- **Gate: OFFLINE** (gray, gray dot) -- the network gate is closed. Only
  localhost connections are allowed.

**Clicking the Gate label** toggles the mode, same as the title bar buttons.
If you click it to go online and credentials are missing, you get the same
warning dialog.

---

## 7. Admin Menu

Open it from the menu bar: **Engineering > Admin Settings...**

This opens a separate child window with four sections and two buttons at
the bottom. Changes take effect immediately -- there is no "Apply" or
"Save" button. The config object in memory updates the moment you move a
slider.

### Retrieval Settings

These control how the system searches your indexed documents.

**top_k** (slider, 1 to 50, default: 12)

How many document chunks to retrieve and send to the LLM. Higher values
give the LLM more context but cost more tokens and can dilute relevance.

- Leave at 12 for most use cases.
- Lower to 5 if answers are wandering off topic (too much noise).
- Raise to 20-30 if the answer needs information scattered across many
  documents.

**min_score** (slider, 0.00 to 1.00, default: 0.10)

Minimum relevance score a chunk must have to be included. Chunks below
this threshold are discarded even if they are in the top_k.

- 0.10 is permissive (keeps almost everything that matched at all).
- 0.30 is moderate (filters out weak matches).
- 0.50 is strict (only highly relevant chunks survive).
- If you get "no results found" on queries that should work, lower this.
- If answers include irrelevant information, raise this.

**Hybrid search** (toggle, default: ON)

When ON, the system runs two searches in parallel:
- **Vector search:** Finds chunks by meaning similarity. Catches synonyms
  and paraphrases ("operating frequency" matches "radio band").
- **Keyword search (BM25):** Finds chunks by exact words. Catches part
  numbers, acronyms, and specific terms.

The results are merged using Reciprocal Rank Fusion. Leave this ON unless
you have a specific reason to disable one search type.

**Reranker** (toggle, default: OFF)

When ON, a cross-encoder model re-scores the top retrieved chunks for more
accurate relevance ranking.

**Leave this OFF for general use.** Testing showed that enabling the reranker
destroys accuracy on unanswerable questions (drops from 100% to 76%),
injection detection (100% to 46%), and ambiguous questions (100% to 82%).
Only enable it if you are doing a narrow, precision-focused search where you
know every question has an answer in the documents.

### LLM Settings

These control how the language model generates answers.

**Max tokens** (slider, 256 to 4096, default: 2048)

Maximum number of tokens the LLM can generate in its answer. One token is
roughly 3/4 of a word.

- 512: Short, focused answers (good for factual lookups).
- 2048: Medium-length answers (good default for most questions).
- 4096: Long answers (for detailed explanations or summaries).

Higher values do not make the model write more -- they just raise the ceiling.
The model stops when it finishes its answer regardless of this limit.

**Temperature** (slider, 0.00 to 1.00, default: 0.05)

Controls randomness in the LLM output.

- 0.00-0.10: Deterministic. Same question gives nearly the same answer
  every time. Best for factual document search.
- 0.20-0.40: Slightly creative. Good for drafting and summarization.
- 0.50+: More varied output. Avoid for factual queries -- the model may
  start inventing details.

The system default is 0.05 (nearly deterministic). This is correct for RAG
where you want the model to stick to what the documents say.

**Timeout (s)** (slider, 10 to 120, default: 30)

How many seconds to wait for the LLM to respond before giving up.

- 30 seconds is enough for online API calls.
- For offline mode on CPU, Ollama can take 30-180 seconds for a single
  answer. If you get timeout errors in offline mode, raise this to 120.

### Performance Profile

A dropdown with three options:

**laptop_safe** (default)
- Embedding batch size: 16
- Search top_k: 5
- Index block: 200K chars
- Concurrent files: 1
- For machines with 8-16 GB RAM.

**desktop_power**
- Embedding batch size: 64
- Search top_k: 10
- Index block: 500K chars
- Concurrent files: 2
- For machines with 32-64 GB RAM.

**server_max**
- Embedding batch size: 128
- Search top_k: 15
- Index block: 1M chars
- Concurrent files: 4
- For overnight runs on 64+ GB machines.

When you change the profile, the system calls `_profile_switch.py` which
writes the new settings directly to `config/default_config.yaml`. This is
one of the few settings that persist to disk.

After switching profiles, existing indexed data still works fine. Only
indexing speed changes.

### Test Query

A small section at the bottom for testing how your current settings affect
answers.

- **Text field:** Type a test question.
- **Run Test button:** Sends the query with current settings.
- **Result area:** Shows the answer (truncated to 500 characters).
- **Latency label:** Shows how long the query took.
- **Network indicator:** If in online mode, shows "Network: ACTIVE" in
  amber while the test query is running. This reminds you that data is
  being sent to the API.

This lets you tweak a slider, immediately run a test, and see the effect
without leaving the engineering menu.

### Reset to Defaults

Restores all sliders and toggles to the values they had when you opened the
engineering menu. Does not reset to the original YAML defaults -- it resets
to whatever the values were when the window was created.

### Close

Closes the engineering menu window. Your changes are already applied.

---

## 8. Menu Bar

### File

- **Exit** -- Closes the application. The status bar timer is stopped
  cleanly.

### Admin

- **Admin Settings...** -- Opens the engineering menu described in
  section 7.

### Help

- **About** -- Shows a dialog with the application name, description, and
  technology stack.

The menu bar itself follows the active theme: dark background with white
text in dark mode, light background with black text in light mode. The
active (hovered) menu item uses the blue accent color (#0078d4).

---

## 9. Dark Mode and Light Mode

The GUI ships with two visual themes defined in `src/gui/theme.py`. Dark
mode is the default. Toggle between them with the Theme button in the
title bar (top right).

### When to use which

- **Dark mode**: Default. Best for daily use, low-light environments, and
  reducing eye strain during long sessions. Most screen elements use dark
  grays with white text.
- **Light mode**: Best for projector demos, brightly lit offices, and
  screenshots for documentation. Uses white/light gray backgrounds with
  black text.

### Typography

All text uses **Segoe UI** (the Windows 11 system font):

| Style | Font | Used for |
|-------|------|----------|
| Regular | Segoe UI 10pt | Most text, labels, buttons |
| Bold | Segoe UI 10pt bold | Panel titles, active labels |
| Title | Segoe UI 13pt bold | Application title |
| Small | Segoe UI 9pt | Test query area, secondary info |

### Color Reference

Every color in the GUI is defined by the active theme. The accent color
(#0078d4, Windows blue) is the same in both themes.

| Element | Dark Mode | Light Mode |
|---------|-----------|------------|
| Main background | #1e1e1e (near-black) | #f0f0f0 (light gray) |
| Panel background | #2d2d2d (dark gray) | #ffffff (white) |
| Primary text | #ffffff (white) | #000000 (black) |
| Secondary text | #a0a0a0 (gray) | #555555 (medium gray) |
| Input field background | #3c3c3c (charcoal) | #ffffff (white) |
| Input field text | #ffffff (white) | #000000 (black) |
| Accent (buttons, links) | #0078d4 (blue) | #0078d4 (blue) |
| Accent hover | #106ebe (darker blue) | #106ebe (darker blue) |
| Border lines | #555555 (gray) | #cccccc (light gray) |
| Disabled text | #777777 (dim gray) | #999999 (dim gray) |
| Success indicator | #4caf50 (green) | #388e3c (dark green) |
| Error indicator | #f44336 (red) | #d32f2f (dark red) |
| Warning indicator | #ff9800 (orange) | #f57c00 (dark orange) |
| Active button (ON) | #4caf50 (green bg) | #388e3c (green bg) |
| Inactive button (OFF) | #3c3c3c (gray bg) | #e0e0e0 (light gray bg) |
| Menu background | #2d2d2d (dark gray) | #f0f0f0 (light gray) |
| Scrollbar track | #3c3c3c (charcoal) | #e0e0e0 (light gray) |
| Scrollbar thumb | #666666 (medium gray) | #b0b0b0 (gray) |
| Separator lines | #555555 (gray) | #cccccc (light gray) |

### Status Indicator Colors

These are consistent across both themes (the indicator colors are the same,
only the surrounding backgrounds change):

| Indicator | Color | Meaning |
|-----------|-------|---------|
| Green dot/text | #4caf50 / #388e3c | Active, ready, success |
| Gray dot/text | #a0a0a0 / #757575 | Inactive, offline, neutral |
| Red text | #f44336 / #d32f2f | Error, failure |
| Orange text | #ff9800 / #f57c00 | Warning, in-progress caution |

### How the Theme Switch Works

When you click the Theme button:

1. The theme dictionary switches from `DARK_THEME` to `LIGHT_THEME` (or
   vice versa).
2. Every widget in the window has its colors updated in place -- no window
   rebuild, no flicker.
3. The menu bar is rebuilt with new background and text colors.
4. The ttk styles (used by comboboxes, progress bars) are reapplied.
5. If the Admin Menu window is open, it updates too.
6. The Theme button label flips: "Light" becomes "Dark" and vice versa.

The theme preference is not persisted to disk. On next launch, the GUI
always starts in dark mode.

---

## 10. How to Run Your First Query

Step by step, starting from nothing:

1. **Have documents indexed.** If you have not indexed yet, follow section
   11 first. You cannot query without an index.

2. **Launch the GUI.** Run `python src/gui/launch_gui.py` and wait for the
   window to appear. The status bar will show "LLM: Not initialized" until
   backends finish loading.

3. **Wait for backends.** Watch the terminal where you launched the GUI.
   When you see `[OK] Backends attached to GUI`, the system is ready. The
   status bar will update to show the active model.

4. **Pick your use case.** Click the Use case dropdown and select the role
   that matches your question. For technical document search, leave it on
   "Engineering / STEM".

5. **Check your mode.** Look at the title bar. If you want to use the
   local Ollama model, make sure OFFLINE is green. If you want to use the
   cloud API, click ONLINE (you need stored credentials first).

6. **Type your question.** Click in the question box, type your question in
   plain English, and press Enter or click Ask.

7. **Wait for the answer.** The Ask button grays out and a status message
   appears. Online answers typically arrive in 3-15 seconds. Offline answers
   on CPU can take 1-3 minutes.

8. **Read the answer.** The answer appears in the scrollable text area.
   Below it, check the Sources line to see which documents contributed.
   Check the Metrics line for latency and token counts.

9. **Ask another question.** The query box clears its previous text when
   you click it. Type a new question and press Enter.

---

## 11. How to Index a New Folder

Step by step:

1. **Gather your documents** into a single folder. Supported formats:
   PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, JSON, XML, LOG, EML.
   Subfolders are included automatically (recursive scan).

2. **Launch the GUI** if it is not already running.

3. **Wait for backends** to finish loading (see terminal output).

4. **Set the source folder.** In the Index Panel, click **Browse** and
   select your documents folder. Or type the path directly.

5. **Click Start Indexing.** The progress bar appears and the system begins
   processing files one at a time.

6. **Watch progress.** The label shows the current file name. The counter
   shows "X / Y files". The progress bar advances after each file.

7. **Wait for completion.** When done, the Last run label updates:
   `Last run: 2026-02-21 14:30 | 4,335 chunks indexed | 287s`

8. **You can now query.** Go to the Query Panel and ask questions about
   your documents.

**Re-indexing:** If you add new files to the folder and run indexing again,
only the new and changed files are processed. Unchanged files are skipped
automatically (hash comparison). This makes updates fast.

**Stopping:** Click Stop to halt after the current file. You can resume
later by clicking Start Indexing again -- it picks up where it left off.

---

## 12. Online vs Offline Mode

### What changes when you go online

| Aspect             | Offline                    | Online                           |
|--------------------|----------------------------|----------------------------------|
| LLM location       | Ollama on localhost        | Cloud API (Azure, OpenRouter)    |
| Network access     | None (localhost only)      | Configured endpoint only         |
| Answer speed       | 30s-3min (CPU dependent)   | 3-15 seconds                     |
| Data sent          | Nothing leaves machine     | Question + top chunks sent       |
| Cost               | Free (local compute)       | Per-token API pricing            |
| Requires           | Ollama running             | API key + endpoint stored        |
| Gate indicator     | Gray "OFFLINE"             | Green "ONLINE"                   |

### What to check before going online

1. **Credentials stored?** You need both an API key and an endpoint URL
   saved in Windows Credential Manager. Run these from PowerShell if you
   have not already:
   ```
   rag-store-key       # Prompts for API key (hidden input)
   rag-store-endpoint  # Prompts for endpoint URL
   ```

2. **Network available?** You need internet access to reach the API. On a
   corporate network, check that your endpoint is not blocked by the
   firewall.

3. **Budget awareness.** Online queries cost money. The Metrics line shows
   token counts so you can estimate cost. At typical OpenAI pricing:
   - 1,000 tokens in: ~$0.0025
   - 1,000 tokens out: ~$0.01
   A single query with 5 chunks typically uses 400-800 tokens in and
   100-300 tokens out.

### What never gets sent (even in online mode)

- Full documents (only small chunk excerpts are sent)
- File paths or file names from your machine
- The SQLite database
- Embedding vectors
- Any files outside the retrieved chunks

### What goes through the network gate

The gate is an application-level firewall. In offline mode it blocks every
outbound connection except localhost. In online mode it allows only the
single configured API endpoint. Everything else is still blocked. The
HuggingFace model hub, telemetry endpoints, and all other external services
remain blocked in both modes.

---

## 13. Keyboard Shortcuts

The GUI responds to the following keyboard shortcuts.

### Global shortcuts (work from any panel)

| Shortcut | Action |
|----------|--------|
| **Tab** | Move focus to the next widget (standard Windows navigation) |
| **Shift+Tab** | Move focus to the previous widget |
| **Alt+F4** | Close the application |
| **Escape** | Close the Admin Menu window (if open) |

### Query Panel shortcuts

| Shortcut | Action |
|----------|--------|
| **Enter** (in question box) | Submit the query (same as clicking Ask) |
| **Ctrl+C** (in answer area) | Copy selected text |
| **Ctrl+A** (in answer area) | Select all answer text |

### Admin Menu shortcuts

| Shortcut | Action |
|----------|--------|
| **Escape** | Close the Admin Menu window |
| **Tab / Shift+Tab** | Navigate between sliders and controls |

**Note:** There is no global hotkey to open the Admin Menu from the
keyboard. Use the menu bar: **Alt** to activate the menu bar, then arrow
keys to navigate to **Engineering > Admin Settings...** and press
**Enter**.

---

## 14. Common Problems and What They Mean

### "Query engine not initialized. Run boot first."

**What happened:** You clicked Ask before the backends finished loading.

**Fix:** Wait for the terminal to show `[OK] Backends attached to GUI`. On
an 8 GB laptop, this can take 30-60 seconds after the window appears.

### "Indexer not initialized. Run boot first."

**Same cause as above.** Wait for backend loading to complete.

### "No folder selected" or "Folder does not exist"

**What happened:** The source folder field in the Index Panel is empty or
points to a path that does not exist.

**Fix:** Click Browse and select a valid folder.

### Credentials Missing dialog

**What happened:** You clicked ONLINE but have not stored your API key or
endpoint.

**Fix:** Open PowerShell and run:
```
rag-store-key
rag-store-endpoint
```
Then try clicking ONLINE again.

### Ollama: Offline (gray in status bar)

**What happened:** Ollama is not running on localhost:11434.

**Fix:** Open a separate terminal and run `ollama serve`. If Ollama is not
installed, download it from https://ollama.com. After starting Ollama, the
status bar will turn green within 5 seconds (next automatic refresh).

If Ollama is running but still shows Offline, check that no corporate proxy
is intercepting localhost. The `NO_PROXY` environment variable should include
`localhost,127.0.0.1`.

### Gate: OFFLINE but you expected ONLINE

**What happened:** Mode switch failed or was reverted.

**Fix:** Click the ONLINE button in the title bar. If it shows a credential
error, store credentials first. Check the terminal output for gate
reconfiguration messages.

### Answer text shows "[FAIL]" in red

**What happened:** The query failed. Common reasons:

| Error text contains   | Meaning                               | Fix                                    |
|-----------------------|---------------------------------------|----------------------------------------|
| 401 / Unauthorized    | API key is invalid or expired         | Re-run `rag-store-key` with a valid key |
| 404 / NotFound        | Wrong endpoint URL or deployment name | Check endpoint with `rag-cred-status`   |
| 429 / RateLimit       | Too many API requests                 | Wait 30-60 seconds and try again        |
| Timeout               | LLM took too long to respond          | Raise timeout in Admin menu        |
| Connection            | Network issue                         | Check VPN/internet, verify endpoint      |
| SSL / certificate     | Corporate proxy blocking HTTPS        | Install `pip-system-certs`               |
| NetworkBlockedError   | Gate blocked the request              | Check mode -- are you in offline mode?   |

### Indexing hangs or is very slow

**What happened:** On CPU-only machines, embedding is the bottleneck.
Processing 1,000 files can take 5-30 minutes depending on file size and
hardware.

**Fix:** This is normal. The progress bar and file label show that the
system is still working. If you need faster indexing:
- Switch to `desktop_power` or `server_max` profile (Admin menu).
- Use a machine with a GPU (CUDA) -- set `embedding.device: cuda` in config.
- Run indexing overnight.

### "Out of Memory" during indexing

**What happened:** The embedding batch size is too large for your RAM.

**Fix:** Open Admin menu, switch profile to `laptop_safe`. This
reduces batch size from 64/128 to 16 and processes one file at a time.

### Answer seems wrong or cites irrelevant sources

**Possible causes and fixes:**

1. **min_score too low.** Irrelevant chunks sneak in. Open Admin menu
   and raise min_score from 0.10 to 0.25 or 0.30.

2. **top_k too high.** Too many chunks dilute the good results. Lower top_k
   from 12 to 5-8.

3. **Documents not indexed.** The answer can only come from indexed documents.
   If you recently added files, re-run indexing.

4. **Wrong use case.** If you are asking a general knowledge question but the
   use case is set to Software Engineering, the model selection may not be
   optimal. Switch to General AI or the appropriate role.

### Progress bar does not move

**What happened:** The current file is very large (e.g., a 500-page PDF).
The progress bar advances per file, not per page. A single large file can
take several minutes.

**Fix:** Wait. Watch the terminal for processing messages. If you need to
stop, click Stop -- it will finish the current file and then halt.
