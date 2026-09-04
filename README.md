# AI Fact Checker

MUBA Hackathon 2026 · GonkaRouter Track MVP — a web application for news fact-checking based on multi-model concurrent cross-validation.

## Key Features

| Feature | Description |
| --- | --- |
| Dual input modes | Supports news URLs (article body auto-fetched), news plain text, or direct questions (e.g., "Did floods recently hit Nepal?") |
| Web search for evidence | Automatically retrieves related news evidence (titles/snippets/body excerpts) via DuckDuckGo and injects it into the prompt, solving the problem that models cannot browse the web or answer time-sensitive news |
| Multi-model concurrent fact-checking | Concurrently requests multiple base LLMs through the GonkaRouter unified API (default `deepseek-ai/DeepSeek-V4-Flash-0731` and `moonshotai/Kimi-K2.6`) |
| Structured output | The prompt forces the model to output JSON (`score` integer 0-100 + `reasoning` 2-5 sentence analysis) |
| Truth Score | Linear-algebra-based multi-dimensional matrix consensus engine: builds an N×3 score matrix M (fact/logic/source), dynamically generates the weight vector W from each model vector's Euclidean distance to the centroid (Gaussian kernel decay), and synthesizes the final score via the weighted matrix product v = WᵀM |
| Score matrix visualization | `st.table` clearly displays the N×3 score matrix M and the dynamic weight vector W, along with the centroid and the weighted composite vector |
| Core claim breakdown | The model breaks the content into 2-3 core claims, scoring and judging each one (true/exaggerated/false), displayed as a front-end table |
| Misleading-technique warning | Adds the `bias_warning` field to flag clickbait, emotional manipulation, fabricated data, quoting out of context, etc., shown as prominent warning cards (hidden when there is no risk) |
| Logic topology graph | The model outputs `logic_graph` (nodes/edges), rendered with pyvis as a draggable node graph (claim = blue, evidence = green, conflict = red); vis.js is fully inlined with no external CDN dependency; gracefully falls back to plain text when the graph is missing |
| Score comparison chart | `st.bar_chart` visually compares each model's independent scores, reflecting disagreement between models |
| Reasoning Trace | Each model's reasoning trace is shown in collapsible `st.expander` panels |
| Audit credential | Gonka Request IDs are packaged into a "Decentralized Inference Verification Credential (Powered & Audited by GonkaRouter Network)" visual card |
| Report export | One-click download of a complete Markdown (.md) fact-check report (time, scores, score matrix, claim breakdown, reasoning traces, Request IDs) |
| Cache acceleration | `@st.cache_data(ttl=3600)` caches webpage fetching and web search results, so re-checking the same news incurs no network latency |
| Request ID tracking | The Gonka `request_id` of every API call is shown in `st.code` blocks (page bottom + sidebar) |
| Fetch fault tolerance | Anti-crawler User-Agent, timeout control, and automatic prompts to paste the text manually when fetching fails |
| Bilingual interface | One-click switch between Chinese / English in the sidebar; UI text, system prompts, evidence blocks, and model reasoning all switch with the language |

## Tech Stack

- Frontend interaction: Streamlit (with pyvis topology graph rendering, vis.js inlined with no CDN dependency)
- Network requests: httpx (async + `asyncio.gather` concurrency)
- Web scraping: httpx + BeautifulSoup4 (lxml parser)
- Web search: ddgs (DuckDuckGo, no API key required)
- Consensus engine: numpy (linear algebra matrix operations)
- Language: Python 3.10+

## Installation

```bash
# 1. It is recommended to install in a virtual environment with Python 3.10+
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Docker Deployment (Recommended)

No need to install a Python environment on the host; one command builds and runs everything. All Docker-related files are located in the `docker/` directory.

### 1. Build the Image

Run this in the project root (the directory containing `app.py`):

```bash
docker build -f docker/Dockerfile -t ai-fact-checker:v1 .
```

> The first build downloads the base image and installs dependencies (streamlit / numpy / pyvis / ddgs, etc.), taking about 2-5 minutes; subsequent builds only take a few seconds thanks to Docker layer caching.

### 2. Run the Container

```bash
docker run -d --name ai-fact-checker -p 8501:8501 ai-fact-checker:v1
```

Open `http://localhost:8501` in your browser.

### 3. Common Commands

```bash
# Check status and health (healthy means the service is running properly)
docker ps

# View logs (including Streamlit startup info and errors)
docker logs -f ai-fact-checker

# Stop / Start / Restart / Remove the container
docker stop ai-fact-checker
docker start ai-fact-checker
docker restart ai-fact-checker
docker rm -f ai-fact-checker

# Remove the image
docker rmi ai-fact-checker:v1

# Rebuild and replace the container after updating code
docker build -f docker/Dockerfile -t ai-fact-checker:v1 .
docker rm -f ai-fact-checker
docker run -d --name ai-fact-checker -p 8501:8501 ai-fact-checker:v1
```

### 4. Using docker compose (optional, simpler)

```bash
# Build and start in the background (run in the v1 directory)
docker compose -f docker/docker-compose.yml up -d --build

# View status / logs
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f

# Stop and remove the container
docker compose -f docker/docker-compose.yml down
```

### 5. Docker Environment Variables

| Variable | Description | Default |
| --- | --- | --- |
| `GONKA_API_KEY` | Overrides the default API Key built into `app.py` (leave empty to use the built-in default) | empty (uses the built-in value) |

Example:

```bash
docker run -d --name ai-fact-checker -p 8501:8501 -e GONKA_API_KEY=sk-xxxx ai-fact-checker:v1
```

### 6. FAQ (Docker)

- **Port already in use?** Map to a different host port, e.g. `-p 8502:8501`, then visit `http://localhost:8502`;
- **How to apply code changes?** The Docker image is a snapshot; re-run the build command (see the update flow in "Common Commands");
- **Web search unavailable inside the container?** The container uses the host network (NAT) by default, so DuckDuckGo is reachable; if it fails it is usually rate-limiting, and the app degrades gracefully (same behavior as running locally).

## Usage

1. Start the app:

   ```bash
   streamlit run app.py
   ```

2. Open `http://localhost:8501` in your browser;
3. Use "界面语言 / Language" at the top of the sidebar to switch between Chinese and English; confirm the API Key, the base models (pre-filled by default), and the "Enable web search" toggle (recommended on);
4. Paste a news URL, news text, or a factual question in the main input box (e.g., "尼泊尔最近山洪爆发？" / "Did floods recently hit Nepal?"), then click "Start Fact-Check";
5. View the **Truth Score** at the top (`st.metric`, prominently displayed), the overall verdict, the model disagreement, and the **multi-model score comparison bar chart**; expand the "Web search evidence" panel to see the sources the models referenced;
6. Expand each model's panel to view the **Reasoning trace** and the **core claims breakdown table**; if there is a misleading-technique risk, it is shown as a warning card;
7. View the **Gonka Request IDs** in the **Decentralized Inference Verification Credential** card at the bottom, and click the button to download the Markdown fact-check report.

## Configuration

- **API Key**: defaults to `DEFAULT_API_KEY` at the top of `app.py`; can be overridden by the `GONKA_API_KEY` environment variable or the sidebar input;
- **Model list**: modify `DEFAULT_MODELS` at the top of `app.py`; it is recommended to first call `GET https://api.gonkarouter.io/v1/models` to query the actually available model IDs;
- **Timeout and content length**: the `REQUEST_TIMEOUT` / `MAX_TOKENS` / `FETCH_TIMEOUT` / `MAX_CONTENT_CHARS` constants can be adjusted as needed;
- **Cache**: `CACHE_TTL` (default 3600 seconds) controls how long webpage fetching and web search results are cached; failed results are not cached (exceptions are not cached), and empty search results are not cached either;
- **Web search**: the `SEARCH_MAX_RESULTS` (number of results) / `SEARCH_PAGE_FETCH_N` (number of pages deeply fetched) / `SEARCH_PAGE_TIMEOUT` / `SEARCH_SNIPPET_CHARS` constants are tunable; DuckDuckGo occasionally rate-limits, in which case the app automatically degrades to checking based only on the input content.

## Project Structure

```
.
├── app.py            # Main program (data fetching / API calls / consensus computation / UI rendering)
├── requirements.txt  # Python dependency list
├── README.md         # Project documentation (this file)
├── mechanism.md      # Business logic and formula details
├── .dockerignore     # Docker build context exclusion list
└── docker/
    ├── Dockerfile            # Application container image definition
    └── docker-compose.yml    # One-command build + run orchestration
```

## FAQ

- **Webpage fetching failed?** The page will show "Failed to fetch the page. Please paste the text manually." — just paste the news body and retry;
- **Web search returned nothing?** DuckDuckGo may be rate-limiting; the app automatically degrades to checking based only on the input content, so retry later; for more stability you can integrate a key-based search API such as Serper/Tavily;
- **A model request failed?** Single-model failures are isolated and do not affect the other models; the final score is computed only from the models that returned successfully;
- **Request ID is empty?** The gateway puts the tracking ID in the `x-request-id` response header; the app extracts it with the priority `header x-request-id -> body.request_id -> body.id`, see `mechanism.md`;
- **A model does not support JSON mode?** The app automatically degrades (drops the `response_format` field) and retries once.
