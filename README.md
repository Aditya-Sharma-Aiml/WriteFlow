# WriteFlow — AI-Powered Blog Writing Agent

> **From topic to fully-illustrated, research-backed blog post — in one command.**
> An autonomous multi-agent system built with LangGraph, Google Gemini, Pollinations AI, Tavily Search, and Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [How the System Works](#how-the-system-works)
- [Architecture and Workflow](#architecture-and-workflow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Module and Feature Explanation](#module-and-feature-explanation)
- [AI and LLM Components](#ai-and-llm-components)
- [APIs and External Services](#apis-and-external-services)
- [Data Flow](#data-flow)
- [Installation and Setup](#installation-and-setup)
- [Environment Variables](#environment-variables)
- [Running Locally](#running-locally)
- [Usage Walkthrough](#usage-walkthrough)
- [Screenshots and Demo](#screenshots-and-demo)
- [Error Handling and Security](#error-handling-and-security)
- [Performance and Optimization](#performance-and-optimization)
- [Future Improvements](#future-improvements)
- [Limitations](#limitations)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**WriteFlow** is a production-quality, agentic blog-writing system that takes a bare topic string and autonomously produces a complete, publication-ready Markdown blog post — including inline technical diagrams and illustrations, sourced citations, and precise section-level word-count targeting.

The system is built on a **LangGraph state-machine** that orchestrates specialized agent nodes:
- **Router** (classifies topic as `closed_book`, `hybrid`, or `open_book` and sets research scope)
- **Researcher** (fetches live web evidence via Tavily and synthesizes date-normalized sources)
- **Orchestrator** (generates a structured, concise content plan with 3–5 targeted sections)
- **Parallel Workers** (writes each section concurrently via LangGraph `Send` fanout)
- **Reducer Subgraph** (merges sections, plans image placements, and renders visual graphics via Pollinations AI / Gemini)

A responsive **Streamlit frontend** (`bwa_frontend.py`) provides real-time streaming progress, tabbed views (Plan, Evidence, Markdown Preview, Images, and Logs), a past-blog browser, and one-click export of the finished Markdown and image bundle as a ZIP file.

---

## Problem Statement

Writing high-quality technical blog posts is expensive and time-consuming:

| Pain Point | Reality |
|---|---|
| **Research** | Manually hunting for authoritative, up-to-date sources is slow. |
| **Structuring** | A coherent, focused section outline with balanced word budgets takes planning. |
| **Writing** | Producing each section at the right depth with code snippets and citations is mentally taxing. |
| **Illustrations** | Sourcing or generating technical architecture diagrams requires separate tools and effort. |
| **Recency** | "Latest benchmarks" or tool releases from months ago become quickly outdated. |

WriteFlow automates each of these steps within minutes while keeping the human firmly in control of the topic and date context.

---

## Key Features

- **Intelligent routing** — automatically classifies any topic as `closed_book`, `hybrid`, or `open_book` and adjusts the research window accordingly (evergreen vs. 45 days vs. 7 days).
- **Live web research** — runs scoped Tavily queries, deduplicates results by URL, filters by recency cutoff, and synthesizes structured `EvidencePack` items.
- **Resilient Gemini Multi-Model Pool** — dynamically cycles across Google Flash models (`gemini-3.5-flash-lite`, `gemini-3-flash-preview`, `gemini-3.6-flash`) to distribute requests across distinct quota pools and prevent single-model 429/503 rate limits.
- **Structured planning** — the Orchestrator emits a typed `Plan` (via Pydantic v2 schemas) with per-section goals, bullets, target words, and citation/code flags.
- **Parallel section writing** — LangGraph's `Send` fanout writes all sections concurrently; sections are reduced back in order using Python's `operator.add` reducer.
- **100% Free & Fast AI image generation** — plans 1–2 technical diagrams and generates real PNG images via **Pollinations AI (Flux engine)** with Gemini fallback, eliminating free-tier quota limits.
- **One-click export** — download the blog as a standalone `.md` file or as a full bundle ZIP (Markdown + `images/` directory).
- **Past blog library** — sidebar scans and lists previously generated `.md` files; one click reloads any past post into the active preview.
- **Single-pass streaming UI** — live progress updates for nodes, evidence counts, sections written, and image counts without redundant duplicate runs.

---

## How the System Works

```
User enters topic + as-of date
          |
          v
    +-------------+
    |   ROUTER    |  Classifies topic -> closed_book / hybrid / open_book
    +------+------+        Sets recency_days (3650 / 45 / 7)
           |
     needs_research?
     +-----+--------+
    YES             NO
     |               |
     v               |
  +----------+       |
  | RESEARCH |       |  Runs Tavily queries, synthesizes EvidencePack
  +----+-----+       |
       |             |
       +------+------+
              v
    +------------------+
    |  ORCHESTRATOR    |  Emits structured Plan (3-5 focused tasks, bullets, targets)
    +--------+---------+
             |  fanout (Send)
    +--------+--------+
    |  WORKER x N     |  Writes each section in parallel (with citations/code)
    +--------+--------+
             |  reducer (operator.add)
    +--------+------------------------------+
    |  REDUCER SUBGRAPH                    |
    |  merge_content -> decide_images      |
    |    -> generate_and_place_images      |
    +--------------------------------------+
             |
             v
   Final Markdown saved to disk + returned to UI
```

---

## Architecture and Workflow

### Main Graph (LangGraph StateGraph)

```
START -> router -> [research] -> orchestrator -> worker(xN) -> reducer_subgraph -> END
                 (conditional)              (fanout/Send)  (compound node)
```

### Reducer Subgraph (nested StateGraph)

```
merge_content -> decide_images -> generate_and_place_images
```

### State Schema (`State: TypedDict`)

| Field | Type | Description |
|---|---|---|
| `topic` | `str` | User-supplied topic |
| `mode` | `str` | `closed_book` / `hybrid` / `open_book` |
| `needs_research` | `bool` | Router decision |
| `queries` | `List[str]` | Scoped web search queries |
| `evidence` | `List[EvidenceItem]` | Deduplicated, recency-filtered web sources |
| `plan` | `Optional[Plan]` | Structured content plan |
| `as_of` | `str` | ISO date used as recency anchor |
| `recency_days` | `int` | Cutoff window per mode |
| `sections` | `Annotated[List[tuple], operator.add]` | `(task_id, section_md)` pairs, parallel-reduced |
| `merged_md` | `str` | Assembled markdown before images |
| `md_with_placeholders` | `str` | Markdown with `[[IMAGE_1]]`, `[[IMAGE_2]]` tokens |
| `image_specs` | `List[dict]` | Image specifications (prompt, filename, alt, caption) |
| `final` | `str` | Final rendered Markdown with embedded images |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph + Send fanout + subgraphs) |
| **Primary LLM Engine** | [Google Gemini Flash](https://ai.google.dev/) (`gemini-3.5-flash-lite`, `gemini-3-flash-preview`, `gemini-3.6-flash`) via `google-genai` |
| **Fallback LLM Engine** | [OpenRouter](https://openrouter.ai/) / [OpenAI](https://platform.openai.com/) via `langchain-openai` |
| **Structured Output** | Pydantic v2 (`BaseModel`) with JSON Schema validation |
| **Web Research** | [Tavily Search](https://tavily.com/) via `langchain-community` |
| **Image Generation** | [Pollinations AI](https://pollinations.ai/) (Flux model) + Gemini image fallback |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Data / Display** | [Pandas](https://pandas.pydata.org/) |
| **Environment** | `python-dotenv` |
| **Language** | Python 3.10+ |

---

## Project Structure

```
WriteFlow-blog-writing-agent/
|
+-- bwa_backend.py              # Core LangGraph graph, GeminiLLM engine, image generation, compiled app
+-- bwa_frontend.py             # Streamlit UI — streaming, tabs, downloads, past-blog library
+-- requirements.txt            # Project dependencies
+-- .env                        # Local secrets (API keys, not committed)
+-- .gitignore                  # Git exclusions (.env, images, virtualenv, caches)
|
+-- 1_bwa_basic.ipynb           # Notebook 1: Minimal 3-node graph (orchestrator->worker->reducer)
+-- 2_bwa_improved_prompting.ipynb  # Notebook 2: Richer prompts, structured tasks, word targets
+-- 3_bwa_research.ipynb        # Notebook 3: Router + Tavily research node added
+-- 4_bwa_research_fine_tuned.ipynb # Notebook 4: Recency filtering, date-aware evidence synthesis
+-- 5_bwa_image.ipynb           # Notebook 5: Image generation pipeline
+-- tavily_test.ipynb           # Standalone Tavily API sandbox
|
+-- images/                     # Auto-created: generated PNG images per blog run
|   +-- *.png
|
+-- *.md                        # Auto-created: saved blog output files (slug-named)
```

---

## Module and Feature Explanation

### `bwa_backend.py` — Agent Pipeline

#### Pydantic Schemas

| Schema | Purpose |
|---|---|
| `Task` | One blog section: id, title, goal, bullets (3-5), target_words, citation/code flags |
| `Plan` | Full blog plan: blog_title, audience, tone, blog_kind, constraints, task list |
| `EvidenceItem` | One web result: title, URL, published_at, snippet, source |
| `RouterDecision` | Router output: mode, needs_research, queries, max_results_per_query |
| `EvidencePack` | Synthesized list of `EvidenceItem` objects |
| `ImageSpec` | One image: placeholder token, filename, alt, caption, prompt, size, quality |
| `GlobalImagePlan` | Full image plan: updated markdown with placeholders + list of `ImageSpec` |

#### Resilient `GeminiLLM` Multi-Model Pool
- Cycles requests in round-robin fashion across `gemini-3.5-flash-lite`, `gemini-3-flash-preview`, and `gemini-3.6-flash`.
- Because Google AI Studio rate limits are applied per-model (e.g. 5 RPM), distributing requests across models prevents quota saturation.
- Supports both standard text invocation and strict Pydantic JSON schema generation via `response_mime_type="application/json"`.

#### Router Node
Classifies the topic into one of three research modes:
- `closed_book` — evergreen technical concepts (no web search needed, recency window = 3650 days)
- `hybrid` — foundational concepts with recent tools/benchmarks needed (recency window = 45 days)
- `open_book` — weekly news, model releases, volatile pricing/policy (recency window = 7 days)

#### Research Node
Calls Tavily Search across generated queries, deduplicates results by URL, and normalizes published dates to ISO format.

#### Orchestrator Node
Creates a structured outline plan with 3–5 focused sections (target words 150–350 per section) ensuring high editorial quality and fast generation without unnecessary token latency.

#### Fanout & Workers
Dispatches workers in parallel using LangGraph `Send`. Each worker formats its section in Markdown with `## <Section Title>`, adheres strictly to bullet points, incorporates code snippets if flagged, and adds inline `[Source](URL)` citations for external claims.

#### Reducer Subgraph
1. `merge_content`: Assembles workers' sections in ordered sequence.
2. `decide_images`: Analyzes the blog content and inserts 1–2 visual placeholders (`[[IMAGE_1]]`, `[[IMAGE_2]]`).
3. `generate_and_place_images`: Generates images via Pollinations AI (Flux engine), writes `.png` files to `images/`, and embeds them into the Markdown with descriptive alt text and captions.

---

### `bwa_frontend.py` — Streamlit Application

| Feature | Implementation |
|---|---|
| **Sidebar** | Topic input, as-of date selector, Generate button, Past blogs library |
| **Single-pass streaming** | Streams graph updates live using `try_stream()` without duplicate re-invocations |
| **Progress status** | Live node transitions and JSON state indicator (mode, queries, evidence, sections, images) |
| **Plan tab** | Interactive DataFrame of section outlines, goals, bullets, and target words |
| **Evidence tab** | Sourced citations table with clickable source links |
| **Markdown Preview** | Full formatted blog view with rendered inline images and captions |
| **Images tab** | Visual gallery of all generated image assets with file details |
| **Logs tab** | Detailed raw graph execution logs |
| **Downloads** | One-click download buttons for raw `.md` file and MD + images ZIP bundle |

---

## AI and LLM Components

| Component | Model / Engine | Role |
|---|---|---|
| **Router** | Gemini Flash (`gemini-3.5-flash-lite`) | Classifies topic into research mode and generates queries |
| **Evidence Synthesizer** | Gemini Flash (`gemini-3-flash-preview`) | Normalizes raw web search results into typed `EvidencePack` |
| **Orchestrator** | Gemini Flash (`gemini-3.6-flash`) | Generates structured `Plan` with section tasks |
| **Workers** | Gemini Flash Pool (round-robin) | Write sections in parallel with code and citations |
| **Image Planner** | Gemini Flash (`gemini-3.5-flash-lite`) | Inserts image placeholders and crafts image prompts |
| **Image Generator** | Pollinations AI (Flux engine) | Generates high-quality technical diagrams (PNG) |

---

## APIs and External Services

| Service | Purpose | Configuration |
|---|---|---|
| **Google Gemini API** | Primary LLM for routing, planning, and section writing | `GOOGLE_API_KEY` in `.env` |
| **Tavily Search API** | Real-time web search for `hybrid` and `open_book` modes | `TAVILY_API_KEY` in `.env` |
| **Pollinations AI** | Free, fast GPU image generation (Flux engine) | No API key required |
| **OpenRouter / OpenAI** | Secondary LLM fallback if Google key is not present | `OPENROUTER_API_KEY` / `OPENAI_API_KEY` |

---

## Installation and Setup

### Prerequisites

- Python **3.10+**
- Virtual environment (`venv` or `conda`)
- Google Gemini API key (free at [Google AI Studio](https://aistudio.google.com/))
- Tavily API key (free at [Tavily AI](https://tavily.com/))

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/WriteFlow-blog-writing-agent.git
cd WriteFlow-blog-writing-agent

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Primary LLM Engine (Google Gemini Free Tier / Vertex AI key)
GOOGLE_API_KEY=AIzaSy...

# Web Research (Required for hybrid and open_book research modes)
TAVILY_API_KEY=tvly-...

# Optional Fallback LLM Providers
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
```

| Variable | Required | Used By | Notes |
|---|---|---|---|
| `GOOGLE_API_KEY` | **Recommended** | All LLM calls (Router, Orchestrator, Workers, Image Planner) | Uses Gemini multi-model pool |
| `TAVILY_API_KEY` | For research | Research node | If missing, system falls back to `closed_book` mode |
| `OPENROUTER_API_KEY` | Optional | Fallback LLM | Used when `GOOGLE_API_KEY` is not provided |

---

## Running Locally

### Streamlit Web App (Recommended)

```bash
streamlit run bwa_frontend.py
```

The application will open in your browser at `http://localhost:8501`.

### Programmatic Python Invocation

```python
from datetime import date
from bwa_backend import app

inputs = {
    "topic": "Microservices Architecture with Docker",
    "mode": "",
    "needs_research": False,
    "queries": [],
    "evidence": [],
    "plan": None,
    "as_of": date.today().isoformat(),
    "recency_days": 7,
    "sections": [],
    "merged_md": "",
    "md_with_placeholders": "",
    "image_specs": [],
    "final": "",
}

output = app.invoke(inputs)
print(output["final"])
```

---

## Usage Walkthrough

1. **Launch App:** Start the server with `streamlit run bwa_frontend.py`.
2. **Enter Topic:** Type a subject in the topic input (e.g., *"Event-Driven Architecture in Python"*).
3. **Set As-Of Date:** Defaults to today. Can be adjusted for historical or back-dated research.
4. **Generate Blog:** Click **🚀 Generate Blog**.
5. **Live Progress:** Observe the streaming state (Node name, Evidence count, Sections done, Images).
6. **Preview & Export:**
   - Review the post in the **Markdown Preview** tab.
   - Inspect evidence sources in the **Evidence** tab.
   - View images in the **Images** tab.
   - Click **⬇️ Download Markdown** or **📦 Download Bundle (MD + images)**.

---

## Error Handling and Resilience

- **Multi-Model Quota Failover:** If any Gemini model encounters a 429 or 503 response, `GeminiLLM` automatically retries with the next model in the pool.
- **Fail-Safe Image Generation:** If custom prompt generation takes longer than expected, Pollinations automatically retries with an optimized fallback prompt so that images are always created.
- **Graceful Web Search Degradation:** If `TAVILY_API_KEY` is absent or encounters network timeouts, the research node returns an empty list and the orchestrator writes from internal knowledge.
- **Single-Pass Stream Safety:** Frontend state updates accumulate without re-triggering whole graph executions.

---

## Contributing

Contributions and feedback are welcome!
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
