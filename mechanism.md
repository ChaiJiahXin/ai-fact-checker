# mechanism.md — AI Fact Checker: Business Logic and Computation Details

This document explains the project's business logic, data flow, computation formulas, and fault-tolerance design in detail, for reviewers and users to understand the internal mechanism.

## 1. Overall Architecture and Data Flow

```
User input (URL / plain text / factual question)
   │
   ▼
[1] Input classification ── is it a URL? ──yes──▶ Webpage fetching (httpx async + browser UA)
   │                                              │
   │                                              ▼
   │                                       Body extraction (BeautifulSoup noise removal + body localization)
   │                                              │ failed
   │ no ◀──────────────────────────────────────────┘ prompt "Fetching failed, please paste the text manually"
   ▼
[2] Web search for evidence (DuckDuckGo search → concurrently fetch top N result pages → assemble evidence block)
   │                                              │ no results / rate-limited
   │                                              └──▶ Degrade: check based only on the input content (front-end warning)
   ▼
[3] Structured prompt assembly (SYSTEM_PROMPT + body truncated to 6000 chars + evidence injection)
   │
   ▼
[4] Two-phase model requests
    ├── Round 1: asyncio.gather concurrent requests (POST /chat/completions × N)
    └── Round 2: serially retry the round-1 failed/timed-out models with a halved timeout
   │
   ▼
[5] Response parsing (request_id / JSON extraction / thought_process + 3-D scores / claims / logic_graph validation)
   │
   ▼
[6] Matrix consensus engine (N×3 score matrix → centroid → L2 distance → Gaussian-kernel weights → WᵀM weighted synthesis)
   │
   ├── disagreement d̄ > DISAGREEMENT_THRESHOLD (15.0)?
   │        └─▶ [6b] Meta-Judge arbitration (DEFAULT_MODELS[0] as arbiter → final_score / final_verdict)
   ▼
[7] UI rendering (st.metric / st.table / pyvis topology graph / st.code credential / Meta-Judge ruling card)
```

## 2. Input and Fetching Module

- **URL detection**: regex `^https?://[^\s]+$` (case-insensitive). If matched, proceed to the fetching flow; otherwise treat the input as plain text.
- **Fetching requests**: use `httpx.AsyncClient` for an async GET carrying standard browser `User-Agent`, `Accept`, and `Accept-Language` headers, with `follow_redirects=True` and a 20-second timeout, to cope with anti-crawling.
- **Body extraction** (`extract_article_text`):
  1. Remove noisy tags such as `script/style/noscript/iframe/nav/footer/header/aside/form/button`;
  2. Extract the meta summary (`og:description`/`twitter:description`/`description`) — so that a content overview is still available when the body of paywalled/JS-rendered pages is invisible;
  3. Prefer `<article>`, then `<main>`; otherwise heuristically select the longest candidate container by class keywords (article-content / entry-content / post-content, etc.); finally fall back to `<body>`;
  4. Keep only `<p>` paragraphs with length ≥ 20 characters, filtering out navigation and ad residue; if the body is shorter than 50 characters, fall back to the full text;
  5. The output is concatenated in order: title → summary → body.
- **Exception handling**: fetch timeout, abnormal HTTP status (4xx/5xx), parsing failure, and empty body all raise `RuntimeError`, caught upstream and shown to the user as "Fetching failed, please paste the text manually".

## 3. Web Search Module (solving the problem that LLMs cannot browse the web)

LLMs have knowledge cutoffs and cannot browse the web, so they cannot give definitive answers to time-sensitive questions such as "Did floods recently hit Nepal?". This module automatically retrieves evidence before checking and injects it into the prompt:

1. **Search-query construction**:
   - Text/question mode: take the first 120 characters of the input and strip trailing punctuation (`？?!！。，`);
   - URL mode: prefer the page title (the `Title: ` line); if missing, take the first 80 characters of the body.
2. **Search engine**: `ddgs` (DuckDuckGo, no API key required), `DDGS().text(query, max_results=SEARCH_MAX_RESULTS)` with `SEARCH_MAX_RESULTS = 6`, executed in a separate thread via `asyncio.to_thread` to avoid blocking the event loop.
3. **Evidence deepening**: for the top `SEARCH_PAGE_FETCH_N` (default 2) results, concurrently fetch the page bodies with httpx (reusing the browser UA headers, 20-second timeout), each excerpt capped at `SEARCH_SNIPPET_CHARS` (default 3000) characters; individual failures are silently skipped.
4. **Evidence block format**: each source contains `Title / URL / Snippet / Body excerpt`, injected into the user message marked as "[Web Search Evidence]"; the system prompt requires the model to use the evidence as the primary basis for judgment.
5. **Degradation strategy**: when search is rate-limited or returns nothing, an empty evidence block is returned and the front end warns "Web search returned no results; checking based only on the input content this time", while the main flow is not interrupted; if `ddgs` is not installed, an installation hint is shown and the search is skipped.
6. **Transparency**: the evidence actually used can be viewed in the "Web search evidence used in this check" collapsible panel on the results page.

## 4. Two-Phase Multi-Model Request Module (degraded retry for unstable APIs)

- **Round 1 — concurrent requests**: `asyncio.gather(*tasks, return_exceptions=True)` sends N model requests to GonkaRouter at once; `httpx.AsyncClient` reuses a single connection pool. A single model's timeout (`REQUEST_TIMEOUT = 150` seconds) or exception is isolated and affects only itself; the others return normally.
- **Round 2 — serial retry**: after round 1, models with `ok == False` (failed or timed out) are retried one by one in **serial order** — avoiding re-triggering the provider's concurrency limit — with the timeout **halved** (`REQUEST_TIMEOUT / 2`). A successful retry is backfilled into the final results list, which is then handed to the matrix consensus engine.
- **Timeout control**: every model request (and every retry) has its own independent timeout so a slow model cannot block the whole flow; special tasks (e.g. Meta-Judge) can pass a custom `timeout`.

## 5. Structured Prompt Design

- **System prompt**: positions the model as a "rigorous professional fact-checker", examining four dimensions:
  1. Factual consistency (whether the content matches established facts / common sense)
  2. Logical soundness (whether the argument leaps or contradicts itself)
  3. Source credibility (whether citations, data, and attributions are suspicious)
  4. Misleading techniques (exaggerated headlines, quoting out of context, emotional manipulation, fabricated data)
- **Chain of thought (`thought_process`)**: before outputting the final scores, the model **must** first analyze the factual evidence and logical deduction step by step in the `thought_process` field; it is strictly forbidden to reach a conclusion first and then fit the scores to it. This makes each model's reasoning transparent and auditable.
- **Output constraints**: both the system prompt and the tail of the user message emphasize "output only a single JSON object"; also `max_tokens=3000` is set (to accommodate the thought process + claims + logic_graph), preventing the model from emitting verbose Markdown under the evidence prompt, which would slow generation or even time out;
- **Output structure upgrade** (JSON schema):
  - `thought_process`: step-by-step reasoning about the factual evidence and logic, produced before the scores;
  - `score` (integer 0-100): overall truthfulness score;
  - `fact_score` / `logic_score` / `source_score` (integers 0-100): three independent dimension scores — factual consistency, logical soundness, source reliability — used by the matrix consensus engine to build the N×3 score matrix;
  - `reasoning`: 2-5 sentence overall analysis;
  - `claims`: an array of 2-3 core claims, each item containing `claim` (claim text), `verdict` (one of true/exaggerated/false), `score` (integer 0-100), enabling fine-grained breakdown scoring;
  - `bias_warning`: misleading-technique risk note (clickbait/emotional manipulation/fabricated data/quoting out of context), empty string when there is no risk;
  - `logic_graph`: the logic topology graph structure — `nodes` (id, label, group ∈ {claim/evidence/conflict}), `edges` (source, target, label ∈ {supports/refutes}); source/target must reference existing nodes;
- **Output-language enforcement**: the prompt explicitly requires — regardless of the language of the input content, all output fields (including `thought_process`) must be in Chinese in Chinese mode and in English in English mode;
- **Backward compatibility**: when the model omits `thought_process`/`claims`/`bias_warning`/`logic_graph`/the three dimension scores, `dict.get(..., [])` / `get(..., "")` / falling back to the overall score provide safe degradation; invalid entries are validated and dropped one by one and never crash the system; on JSON parsing failure there is a two-level rescue of trailing-comma repair and regex fallback extraction.
- **Date injection**: today's date (`datetime.date.today()`) is dynamically injected at the start of the system prompt to prevent the model from misjudging recent events as "future events" due to its knowledge cutoff — critical for time-sensitive fact-checking;
- **Evidence priority**: if the user message contains [Web Search Evidence], the model must use it as the primary basis for judgment; if the evidence is insufficient or contradictory, state so honestly and give a conservative score;
- **Bilingual prompts**: when the interface language switches, the system prompt, user-message template, evidence-block labels, and verdict text all switch (Chinese / English); the JSON output-format constraints stay identical in both languages, without affecting score/reasoning parsing;
- **JSON parsing tolerance**: besides strict parsing, trailing-comma repair and regex fallback extraction (`extract_score_reasoning`, which also rescues `thought_process`) are provided to rescue as much as possible from format flaws in the model output.
- **Low randomness**: `temperature=0.1` keeps scores stable and reproducible.
- **JSON-mode degradation**: the first request carries `response_format={"type":"json_object"}`; if the model does not support it (returns 400), the field is automatically dropped and one retry is made.
- **Body truncation**: the body is truncated at 6000 characters to avoid exceeding the model's context window.

## 6. JSON Parsing Algorithm (`extract_json_block`)

Model output may contain extra text, so it is robustly extracted with a three-level strategy:

1. **Direct parsing**: full `json.loads` parse;
2. **Fence stripping**: regex-remove ` ```json ... ``` ` code fences and retry;
3. **Bracket matching**: from the first `{`, do a depth count (correctly handling quotes and escapes inside strings) and extract the first complete JSON object for parsing.

**Fallback extraction** (`extract_score_reasoning`): when the JSON still cannot be fully parsed, regex extracts `score` / `fact_score` / `logic_score` / `source_score` / `thought_process` / `reasoning` directly, rescuing the call as much as possible.

**score validation**: `round(float(score))` is clamped to the `[0, 100]` range after rounding; non-numeric or missing values mark that model as failed. When `score` is missing, `call_model` falls back to `final_score`, which keeps the model compatible with special tasks such as Meta-Judge.

## 7. Matrix Consensus Engine (based on linear algebra)

Each successful model outputs a three-dimensional feature vector $m_i = [\text{fact\_score}, \text{logic\_score}, \text{source\_score}]$. Let N be the number of valid models. The final Truth Score is synthesized as follows:

**Step 1: Build the N×3 score matrix M**

\[
M = \begin{bmatrix} m_1 \\ m_2 \\ \vdots \\ m_N \end{bmatrix} \in \mathbb{R}^{N \times 3}
\]

**Step 2: Compute the centroid (mean vector)**

\[
c = \frac{1}{N} \sum_{i=1}^{N} m_i
\]

(Column-wise mean of the matrix: `M.mean(axis=0)`)

**Step 3: Compute the Euclidean distance (L2 norm) of each feature vector to the centroid**

\[
d_i = \| m_i - c \|_2 = \sqrt{\sum_{j} (m_{ij} - c_j)^2}
\]

**Step 4: Dynamic weights (Gaussian-kernel decay + L1 normalization)**

\[
w_i = \exp\left( -\frac{d_i^2}{2 \sigma_d^2} \right), \quad \sigma_d = \mathrm{std}(d) \;(\text{standard deviation of the distance distribution})
\]

\[
W = \frac{w}{\sum w} \quad (\text{ensuring } \textstyle\sum_i w_i = 1)
\]

- The closer a vector is to the centroid, the higher its weight; outlier models (scores deviating from the majority) have their weights decayed exponentially by the Gaussian kernel;
- When $\sigma_d \approx 0$ (all models highly consistent, or only one model), it degrades to equal weights, ensuring numerical stability;
- Mathematical property: when N=2, the centroid is exactly the midpoint and the two distances are necessarily equal, so the weights degenerate to [0.5, 0.5] — the correct result of geometric symmetry.

**Step 5: Weighted matrix product synthesis**

\[
v = W^T M
\]

(The 1×3 weighted composite vector; numpy: `weights @ M`)

**Step 6: Final Truth Score and disagreement**

\[
\mathrm{Truth\ Score} = \mathrm{mean}(v) = \frac{1}{3} \sum_j v_j
\]

(rounded to 1 decimal place)

\[
\bar{d} = \frac{1}{N} \sum_i d_i
\]

(average Euclidean distance)

The smaller $\bar{d}$ is, the stronger the consensus among models; the larger it is, the more the models disagree on the same content, signaling the user to be cautious before trusting it. When $\bar{d}$ exceeds `DISAGREEMENT_THRESHOLD` (default 15.0), the system automatically launches the Meta-Judge arbitration module (see Section 8).

**Step 7: Verdict classification table** (graded by Truth Score)

| Truth Score | Verdict |
| --- | --- |
| ≥ 80 | Highly credible |
| 60 – 79 | Mostly credible |
| 40 – 59 | Questionable |
| < 40 | Highly suspicious |
| no valid score | Undetermined |

The front end displays the N×3 matrix M (fact/logic/source) and the weight column W with `st.table`, along with numeric annotations of the centroid c and the weighted composite vector v; if all models fail, it shows "Undetermined" and prompts the user to check the API Key / model IDs.

## 8. Meta-Judge (Arbiter) Arbitration Module

When models strongly disagree on the same content, a weighted average is no longer meaningful. The system therefore adds an arbiter (`Meta-Judge`) for a second ruling:

- **Trigger condition**: $\bar{d} >$ `DISAGREEMENT_THRESHOLD` (default 15.0) and at least one model succeeded.
- **Arbiter model**: `DEFAULT_MODELS[0]` (default DeepSeek), invoked through the same `call_model` with a dedicated bilingual system prompt (`META_JUDGE_PROMPT_ZH` / `META_JUDGE_PROMPT_EN`).
- **User message construction** (`build_meta_judge_message`): concatenates the content being fact-checked, the current web-search evidence, and each successful model's scores (fact/logic/source), `thought_process`, and `reasoning`, so the arbiter can weigh every viewpoint.
- **Output structure**: the arbiter outputs `summary` (synthesis of viewpoints and core disagreements), `final_verdict` (credible / questionable / false with reasons), and `final_score` (integer 0-100). `call_model` falls back from `score` to `final_score` so the arbiter's result is extracted correctly.
- **Rendering**: when disagreement exceeds the threshold, a prominent high-disagreement warning is shown at the top of the results area, followed by the Meta-Judge ruling card (arbiter model, final verdict, arbiter score, viewpoint summary, and its own chain of thought). If the arbiter call fails, a warning prompts the user to compare each model's chain of thought and scores manually.

## 9. Request ID Extraction Mechanism

The unique tracking ID of each API call is extracted with the following priority:

1. The response header `x-request-id` (the tracking ID actually returned by the Gonka gateway, e.g. `req-1787990265754132592-378333`);
2. The `request_id` field in the response body;
3. The `id` field in the response body (OpenAI-compatible fallback; note that this field is the chat.completion sequence number, not the gateway tracking ID).

The extracted result is displayed as JSON via `st.code` at the bottom of the page and in the sidebar (required by the competition), and packaged into a "Decentralized Inference Verification Credential (Powered & Audited by GonkaRouter Network)" visual card, emphasizing the system's technical compliance and auditability.

## 10. Visualization, Report, and Caching Mechanisms

- **High-disagreement warning and arbitration card**: when $\bar{d}$ exceeds the threshold, an `st.error` warning plus a bordered Meta-Judge ruling card are shown at the top of the results;
- **Score matrix and weight table**: `st.table` shows the N×3 score matrix M and the dynamic weights W, along with the numeric values of the centroid c and the weighted composite vector v, making the matrix-consensus computation fully transparent;
- **Multi-model score comparison bar chart**: `st.bar_chart` plots the mean of each model's three dimension scores (failed models are recorded as 0 with a note); the height differences intuitively reflect inter-model disagreement, corroborating the d̄ value;
- **Chain-of-thought display**: each model's `thought_process` is shown under a "Chain of Thought" heading inside its collapsible panel and in the exported report, so the reasoning behind every score is fully auditable;
- **Logic reasoning topology graph**: the model's `logic_graph` (nodes/edges) is validated by `parse_logic_graph` (node deduplication, dangling-edge removal, label truncation), then rendered as a draggable dynamic node graph with **pyvis** (`cdn_resources="in_line"`) — vis.js is fully inlined into the HTML and does not reference any external CDN (the template's external Bootstrap CSS link is also removed), so it renders reliably even on restricted networks; node colors are distinguished by group (claim=blue, supporting evidence=green, refutation/conflict=red). When the graph structure is invalid, `None` is returned and the front end gracefully degrades to plain-text reasoning, never erroring out;
- **Misleading-technique warning**: when a model's `bias_warning` is non-empty, it is shown as an `st.warning` card; nothing is shown when there is no risk;
- **Markdown report export**: `build_report()` automatically assembles the check time, composite score, score matrix and weights, weighted composite vector, core claim breakdown, chains of thought + reasoning traces, bias warnings, and Gonka Request IDs, downloadable with one click via `st.download_button`;
- **Caching**: `fetch_webpage_cached` and `get_evidence_cached` use `@st.cache_data(ttl=3600)`. Exceptions raised by failed fetches are not cached by Streamlit (so retries re-fetch); when search returns nothing, `EvidenceNotFoundError` is raised to bypass the cache, avoiding caching a rate-limit-induced empty result for an hour.

## 11. Error Handling Matrix

| Scenario | Handling | User-visible behavior |
| --- | --- | --- |
| Empty input | Intercepted directly | "Please enter a news URL or news text" |
| Missing API Key / no model selected | Intercepted directly | Corresponding error message |
| Webpage fetch timeout / abnormal status / no body | Exception caught | "Fetching failed, please paste the text manually" + technical detail |
| Web search rate-limited / no results | Return empty evidence, degrade to input-content-only checking | Warning "Web search returned no results" |
| ddgs not installed | Skip the search, main flow continues | Warning "ddgs is not installed" |
| Single evidence page fetch failure | Silently skipped, remaining evidence used normally | Not noticeable |
| Round-1 model timeout / exception | Marked failed; round 2 retries it serially with a halved timeout | If the retry succeeds, the result is backfilled silently; if it still fails, the panel shows the error |
| API returns non-200 | Read `error.message` | Panel shows the error code and message |
| Model output is not JSON | JSON extraction fails, that model abstains | Panel shows the parse error + raw output |
| score missing / non-numeric | Fall back to `final_score` (Meta-Judge), otherwise that model abstains | Panel shows the error when no score is available |
| Three dimension scores missing | Fall back to the overall score (the matrix can still be built) | Not noticeable |
| thought_process missing | Omitted from the panel/report, no error | Not noticeable |
| logic_graph structure invalid | Return None, front end degrades to plain-text reasoning | Panel shows the degradation notice |
| Disagreement d̄ > threshold | Launch Meta-Judge arbitration | Warning + Meta-Judge ruling card (or "Meta-Judge failed, compare manually") |
| All models fail | Consensus computation returns "Undetermined" | Error message in the main area |
| Model does not support response_format | Automatically degrade and retry once | Not noticeable |
