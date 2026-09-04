# -*- coding: utf-8 -*-
"""
======================================================================
 AI Fact Checker (AI 事实核查器) —— MUBA Hackathon 2026 · GonkaRouter
======================================================================

[Overview]
    Input a news URL or plain text -> auto-fetch the webpage body (if URL) ->
    search the web for evidence -> concurrently request multiple base LLMs via
    the GonkaRouter unified API; each model first outputs a chain of thought
    (thought_process) and then the three independent dimension scores
    (fact_score / logic_score / source_score), a core-claim breakdown, a
    misleading-technique warning, and a logic topology graph (logic_graph);
    for unstable APIs it automatically degrades to sequential serial retries;
    when the disagreement between models exceeds a threshold, it automatically
    launches an arbiter (Meta-Judge) for a second ruling.
    Underneath it uses a linear-algebra-based multi-dimensional matrix consensus
    engine (numpy):
        - Build an N×3 score matrix M and compute the centroid (mean vector);
        - Dynamically generate the weight vector W from the Euclidean distance
          (L2 norm) of each model's feature vector to the centroid (Gaussian
          kernel decay, outliers down-weighted);
        - Synthesize the final Truth Score via the weighted matrix product
          v = WᵀM;
        - The front end shows matrix M and weights W with st.table, and renders
          a draggable logic reasoning topology graph with pyvis (vis.js inlined,
          no external CDN dependency; gracefully falls back to plain text when
          no valid graph structure exists).
    Supports Chinese / English bilingual UI and prompts, switchable with one
    click in the sidebar.

[Local run]
    1. Install dependencies : pip install -r requirements.txt
    2. Start the app         : streamlit run app.py
    3. Open in browser       : http://localhost:8501

[Module layout] (code is organized into functions by module)
    Data fetching module    : fetch_webpage / fetch_webpage_cached / extract_article_text
    Web search module       : search_web / fetch_evidence_pages / collect_evidence / get_evidence_cached
    API call module         : call_model / run_all_models / run_meta_judge / extract_json_block / parse_claims / parse_logic_graph
    Matrix consensus engine : compute_consensus / classify_verdict (numpy linear algebra)
    Report module           : build_report
    UI rendering module     : render_ui / handle_check / render_results / render_logic_graph
======================================================================
"""

import asyncio
import datetime
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Optional dependency: DuckDuckGo search client (pip install ddgs), used for web-search evidence
try:
    from ddgs import DDGS
    HAS_SEARCH = True
except ImportError:
    DDGS = None
    HAS_SEARCH = False

# Optional dependency: pyvis (pip install pyvis), renders the logic topology graph in in_line mode
# —— vis.js is fully inlined into a self-contained HTML with no external CDN dependency,
#    avoiding blank graphs caused by network interception
try:
    from pyvis.network import Network
    HAS_GRAPH_LIB = True
except ImportError:
    Network = None
    HAS_GRAPH_LIB = False

# ============================ Global configuration ============================
API_BASE_URL = "https://api.gonkarouter.io/v1"
DEFAULT_API_KEY = ""
# Base model list: keep consistent with the real IDs returned by GET {API_BASE_URL}/models
DEFAULT_MODELS = ["deepseek-ai/DeepSeek-V4-Flash-0731", "moonshotai/Kimi-K2.6","MiniMaxAI/MiniMax-M2.7"]

REQUEST_TIMEOUT = 150.0   # Timeout for a single LLM API request (seconds); after evidence injection the prompt is longer and some models respond slower
DISAGREEMENT_THRESHOLD = 15.0  # Disagreement threshold (mean Euclidean distance): when exceeded, show a high-disagreement warning and launch the Meta-Judge arbiter
MAX_TOKENS = 3000         # Cap the max tokens of a single reply (must fit the 3-D scores + claims + logic_graph), preventing verbose Markdown from timing out
FETCH_TIMEOUT = 20.0      # Webpage fetch timeout (seconds)
MAX_CONTENT_CHARS = 6000  # Max characters of the body sent to the model, to avoid exceeding the context window
CACHE_TTL = 3600          # Cache duration for webpage fetching and web search results (seconds)

SEARCH_MAX_RESULTS = 6      # Max number of search results used by the web search
SEARCH_PAGE_FETCH_N = 2     # Number of top results whose bodies are deeply fetched as evidence
SEARCH_PAGE_TIMEOUT = 20.0  # Timeout for fetching a single search-result page (seconds)
SEARCH_SNIPPET_CHARS = 3000 # Max characters of each evidence body excerpt

# The three score dimensions of the matrix consensus engine (matching the JSON fields in the prompt)
DIM_KEYS = ["fact_score", "logic_score", "source_score"]

# Browser User-Agent: mimics a real browser request to reduce the chance of being blocked by anti-crawling
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


@st.cache_data(ttl=60)
def check_api_key(api_key: str) -> tuple[bool, str]:
    """Check whether the Gonka API Key is valid."""
    if not api_key or not api_key.strip():
        return False, "missing"

    try:
        response = httpx.get(
            f"{API_BASE_URL}/models",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )

        if response.status_code == 200:
            return True, "valid"

        if response.status_code in (401, 403):
            return False, "invalid"

        return False, "error"

    except httpx.HTTPError:
        return False, "error"


URL_PATTERN = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)

WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS_EN = ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"]

# Topology graph node colors: claim=blue / evidence=green / conflict=red
GRAPH_GROUP_COLORS = {
    "主张": "#4285F4", "论点": "#4285F4", "claim": "#4285F4",
    "证据": "#33B679", "支撑证据": "#33B679", "evidence": "#33B679", "support": "#33B679",
    "冲突点": "#E74C3C", "反驳": "#E74C3C", "反驳证据": "#E74C3C",
    "conflict": "#E74C3C", "refute": "#E74C3C",
}
GRAPH_DEFAULT_COLOR = "#8C8C8C"

# ============================ Bilingual UI texts ============================
TEXTS: Dict[str, Dict[str, Any]] = {
    "zh": {
        "title": "AI 事实核查器",
        "caption": "多模型并发交叉验证 · 多维矩阵共识引擎 · 一站式新闻真实性核查 · GonkaRouter",
        "lang_label": "界面语言 / Language",
        "sidebar_config": "配置",
        "api_key_label": "Gonka API Key",
        "api_key_connected": "🟢 API Key 已连接",
        "api_key_invalid": "🔴 API Key 无效",
        "api_key_check_error": "⚠️ 无法验证 API Key",
        "models_label": "选择底座模型",
        "extra_models": "追加其他模型 ID（逗号分隔，可选）",
        "enable_search": "启用联网搜索（推荐）",
        "enable_search_help": "输入短问题或时效性新闻时，先联网检索证据再交给模型判断，"
                              "解决大模型无法联网、对最近事件无法作答的问题",
        "no_record": "暂无记录",
        "input_label": "请输入新闻 URL、新闻正文，或需要核查的事实性问题",
        "input_placeholder": "例如：https://example.com/news/123  ·  粘贴一段新闻文字  ·  “尼泊尔最近山洪爆发？”",
        "check_button": "开始核查",
        "warn_empty_input": "请输入新闻 URL 或新闻正文",
        "warn_no_api_key": "请填写 Gonka API Key",
        "warn_no_models": "请至少选择一个底座模型",
        "fetching_page": "正在抓取网页正文……",
        "fetch_failed": "抓取失败，请手动粘贴文本",
        "tech_detail": "技术细节：{}",
        "no_body": "未能从网页中提取到有效正文",
        "no_ddgs": "未安装 ddgs 依赖（pip install ddgs），本次仅基于输入内容核查",
        "searching": "正在联网检索相关证据……",
        "evidence_ok": "已从搜索引擎获取证据并注入模型提示词",
        "search_no_results": "联网搜索未获得结果（可能被限流），本次仅基于输入内容核查",
        "content_too_short": "内容过短，核查结果可能不准确",
        "calling_models": "已提交给 {n} 个模型并发核查，请稍候……",
        "done_caption": "核查完成 · 正文长度 {chars} 字符 · 并发模型数 {models}",
        "all_failed": "所有模型均未能返回有效评分，请检查 API Key / 模型 ID 后重试",
        "metric_score": "Truth Score (0-100%)",
        "metric_verdict": "综合判定",
        "metric_disagree": "模型分歧度 (d̄)",
        "disagree_warning": "模型间存在重大分歧：分歧度 d̄ = {d:.1f} 已超过阈值 {thr:.1f}，已启动仲裁者 (Meta-Judge) 机制进行二次裁决。",
        "meta_judge_running": "检测到模型间重大分歧，正在启动仲裁者 (Meta-Judge) 二次裁决，请稍候……",
        "meta_judge_title": "仲裁结论 (Meta-Judge)",
        "meta_judge_model_label": "仲裁者模型",
        "meta_judge_final": "最终定论",
        "meta_judge_score_label": "仲裁评分",
        "meta_judge_summary": "各方观点总结",
        "meta_judge_failed": "仲裁者调用失败，请手动对比各模型思维链与评分。",
        "meta_judge_no_thought": "（该模型未提供思维链）",
        "consensus_caption": "有效模型 {valid}/{total} 个；Truth Score 由多维矩阵共识引擎加权合成："
                             "对每个模型三维评分向量（事实/逻辑/信源）求质心，以到质心的欧氏距离"
                             "动态生成权重，加权矩阵乘法合成最终得分。",
        "evidence_expander": "本次核查使用的联网检索证据",
        "chart_title": "多模型打分对比",
        "chart_caption": "纵轴为各模型三维评分的平均值；0 分表示该模型未返回有效评分。柱高差异直观反映模型间分歧。",
        "matrix_title": "评分矩阵 M（N×3）与动态权重 W",
        "matrix_caption": "wᵢ ∝ exp(−dᵢ²/2σ²)：模型特征向量越接近质心权重越高，离群值权重高斯衰减；各模型权重之和为 1。",
        "weighted_vector_caption": "质心 c = ({c})；加权合成向量 v = WᵀM = ({v})；Truth Score = mean(v) = {s} / 100",
        "col_fact": "事实一致性",
        "col_logic": "逻辑严谨性",
        "col_source": "信源可靠度",
        "col_weight": "权重 W",
        "bias_label": "误导手法预警",
        "claims_title": "核心断言拆解 (Claims)",
        "col_claim": "断言",
        "col_verdict": "判定",
        "col_score": "评分",
        "reasoning_trace": "推理轨迹 (Reasoning Trace)",
        "graph_legend": "节点颜色：蓝 = 主张，绿 = 支撑证据，红 = 反驳/冲突点（可拖拽交互）",
        "graph_fallback": "该模型未返回有效的图形结构，已优雅降级为纯文本推理展示。",
        "graph_missing_lib": "未安装 pyvis 依赖（pip install pyvis），图形降级为纯文本展示。",
        "score_title": "—— 真实度评分 {score}/100",
        "failed_title": "—— 请求失败",
        "reasoning_label": "推理分析",
        "thought_label": "思维链 (Chain of Thought)",
        "no_reasoning": "（模型未给出推理）",
        "raw_output": "查看模型原始输出",
        "error_label": "错误信息：{err}",
        "unknown_error": "未知错误",
        "credential_title": "去中心化推理验证凭证",
        "credential_network": "Powered & Audited by GonkaRouter Network",
        "request_ids": "Gonka Request IDs",
        "request_ids_caption": "每次 API 调用的唯一追踪 ID，可用于 GonkaRouter 控制台审计与问题排查。",
        "not_obtained": "未获取到",
        "download_report": "下载事实核查报告 (.md)",
        "report_title": "AI 事实核查报告",
        "report_time": "核查时间",
        "report_score": "综合得分",
        "report_verdict": "综合判定",
        "report_disagree": "模型分歧度 (d̄)",
        "report_input": "输入内容",
        "report_matrix": "多维评分矩阵与动态权重",
        "report_vector": "加权合成向量",
        "report_claims": "核心断言拆解",
        "report_trace": "推理轨迹",
        "report_bias": "误导手法预警",
        "report_reqids": "Gonka Request IDs",
        "report_no_bias": "未发现明显误导手法",
        "report_no_claims": "（模型未返回断言拆解）",
        "report_footer": "本报告由多维矩阵共识引擎自动生成 · Powered & Audited by GonkaRouter Network",
        "verdicts": {
            "high": "高度可信", "medium": "基本可信", "low": "存疑",
            "suspect": "高度可疑", "unknown": "无法判定",
        },
        "err_timeout": "请求超时（超过 {t:.0f} 秒）",
        "err_network": "网络请求失败：{e}",
        "err_http": "API 返回 {code}：{msg}",
        "err_structure": "响应结构异常：{e}",
        "err_json": "模型输出无法解析为 JSON",
        "err_score": "score 字段缺失或非数值",
        "err_unhandled": "未捕获异常：{e}",
    },
    "en": {
        "title": "AI Fact Checker",
        "caption": "Multi-model concurrent cross-validation · Matrix consensus engine · One-stop news fact checking · GonkaRouter",
        "lang_label": "界面语言 / Language",
        "sidebar_config": "Settings",
        "api_key_label": "Gonka API Key",
        "api_key_connected": "🟢 API Key Connected",
        "api_key_invalid": "🔴 Invalid API Key",
        "api_key_check_error": "⚠️ Unable to verify API Key",
        "models_label": "Select base models",
        "extra_models": "Add extra model IDs (comma-separated, optional)",
        "enable_search": "Enable web search (recommended)",
        "enable_search_help": "When entering a short question or time-sensitive news, search the web "
                              "for evidence first, solving the problem that LLMs cannot browse the "
                              "web and cannot answer questions about recent events",
        "no_record": "No records yet",
        "input_label": "Enter a news URL, news text, or a factual question to verify",
        "input_placeholder": "e.g. https://example.com/news/123  ·  paste news text  ·  \"Did floods recently hit Nepal?\"",
        "check_button": "Start Fact-Check",
        "warn_empty_input": "Please enter a news URL or news text",
        "warn_no_api_key": "Please fill in the Gonka API Key",
        "warn_no_models": "Please select at least one base model",
        "fetching_page": "Fetching the web page...",
        "fetch_failed": "Failed to fetch the page. Please paste the text manually.",
        "tech_detail": "Technical detail: {}",
        "no_body": "Could not extract valid content from the page",
        "no_ddgs": "ddgs is not installed (pip install ddgs). Checking with input content only.",
        "searching": "Searching the web for evidence...",
        "evidence_ok": "Web evidence retrieved and injected into the model prompt",
        "search_no_results": "Web search returned no results (possibly rate-limited). Checking with input content only.",
        "content_too_short": "Content is very short; the result may be inaccurate",
        "calling_models": "Submitted to {n} models concurrently, please wait...",
        "done_caption": "Check completed · content length {chars} chars · {models} models queried",
        "all_failed": "All models failed to return a valid score. Please check the API Key / model IDs and retry.",
        "metric_score": "Truth Score (0-100%)",
        "metric_verdict": "Verdict",
        "metric_disagree": "Model Disagreement (d̄)",
        "disagree_warning": "Major disagreement detected: d̄ = {d:.1f} exceeds the threshold {thr:.1f}; Meta-Judge arbitration has been triggered.",
        "meta_judge_running": "Major disagreement detected. Launching Meta-Judge arbitration, please wait...",
        "meta_judge_title": "Meta-Judge Ruling",
        "meta_judge_model_label": "Arbiter model",
        "meta_judge_final": "Final ruling",
        "meta_judge_score_label": "Arbiter score",
        "meta_judge_summary": "Summary of viewpoints",
        "meta_judge_failed": "Meta-Judge call failed. Please compare each model's chain of thought and scores manually.",
        "meta_judge_no_thought": "(no chain of thought provided)",
        "consensus_caption": "{valid}/{total} models succeeded; the Truth Score is synthesized by the matrix consensus engine: "
                             "centroid of the 3-D score vectors (fact/logic/source), Gaussian-decay weights by Euclidean "
                             "distance to the centroid, and a weighted matrix product for the final score.",
        "evidence_expander": "Web search evidence used in this check",
        "chart_title": "Model Score Comparison",
        "chart_caption": "Vertical axis shows the mean of each model's three dimension scores; 0 means the model returned no valid score. Height differences reveal model disagreement.",
        "matrix_title": "Score Matrix M (N×3) & Dynamic Weights W",
        "matrix_caption": "wᵢ ∝ exp(−dᵢ²/2σ²): the closer a model's vector is to the centroid, the higher its weight; outlier weights decay. Weights sum to 1.",
        "weighted_vector_caption": "Centroid c = ({c}); weighted vector v = WᵀM = ({v}); Truth Score = mean(v) = {s} / 100",
        "col_fact": "Fact",
        "col_logic": "Logic",
        "col_source": "Source",
        "col_weight": "Weight W",
        "bias_label": "Bias Warning",
        "claims_title": "Core Claims",
        "col_claim": "Claim",
        "col_verdict": "Verdict",
        "col_score": "Score",
        "reasoning_trace": "Reasoning Trace",
        "graph_legend": "Node colors: blue = claim, green = supporting evidence, red = refutation/conflict (draggable)",
        "graph_fallback": "This model returned no valid graph structure; gracefully fell back to plain-text reasoning.",
        "graph_missing_lib": "pyvis is not installed (pip install pyvis); falling back to plain-text display.",
        "score_title": " — Score {score}/100",
        "failed_title": " — Request failed",
        "reasoning_label": "Reasoning",
        "thought_label": "Chain of Thought",
        "no_reasoning": "(no reasoning provided)",
        "raw_output": "View raw model output",
        "error_label": "Error: {err}",
        "unknown_error": "Unknown error",
        "credential_title": "Decentralized Inference Verification Credential",
        "credential_network": "Powered & Audited by GonkaRouter Network",
        "request_ids": "Gonka Request IDs",
        "request_ids_caption": "Unique tracking ID for each API call, for auditing and troubleshooting in the GonkaRouter console.",
        "not_obtained": "not obtained",
        "download_report": "Download Fact-Check Report (.md)",
        "report_title": "AI Fact-Check Report",
        "report_time": "Check time",
        "report_score": "Truth Score",
        "report_verdict": "Verdict",
        "report_disagree": "Model Disagreement (d̄)",
        "report_input": "Input Content",
        "report_matrix": "Score Matrix & Dynamic Weights",
        "report_vector": "Weighted Vector",
        "report_claims": "Core Claims",
        "report_trace": "Reasoning Trace",
        "report_bias": "Bias Warnings",
        "report_reqids": "Gonka Request IDs",
        "report_no_bias": "No obvious misleading techniques detected",
        "report_no_claims": "(model returned no claim breakdown)",
        "report_footer": "This report was auto-generated by the matrix consensus engine · Powered & Audited by GonkaRouter Network",
        "verdicts": {
            "high": "Highly credible", "medium": "Mostly credible", "low": "Questionable",
            "suspect": "Highly suspicious", "unknown": "Undetermined",
        },
        "err_timeout": "Request timed out (>{t:.0f}s)",
        "err_network": "Network error: {e}",
        "err_http": "API returned {code}: {msg}",
        "err_structure": "Unexpected response structure: {e}",
        "err_json": "Model output could not be parsed as JSON",
        "err_score": "score field is missing or not numeric",
        "err_unhandled": "Unhandled exception: {e}",
    },
}


def t(lang: str, key: str) -> Any:
    """Return the text for the current language."""
    return TEXTS[lang][key]


# ============================ Bilingual structured prompts ============================
# Structured prompt: asks the model to output JSON with 3-D scores + claims + bias_warning + logic_graph.
# Note: uses {{today}}/{{weekday}} placeholders + str.replace() for date injection,
# to avoid conflicts between .format() and the braces in the JSON examples.
SYSTEM_PROMPT_ZH = (
    "今天的日期是 {{today}}（{{weekday}}）。"
    "你是一名严谨的专业事实核查员。请根据用户提供的新闻内容，"
    "从以下四个维度综合判断其真实性：\n"
    "1. 事实一致性：内容是否与公认事实、常识相符；\n"
    "2. 逻辑合理性：论证是否存在跳跃或自相矛盾；\n"
    "3. 信息来源可信度：引用、数据、出处是否可疑；\n"
    "4. 误导手法：是否存在夸大标题、断章取义、情绪煽动、虚假数据等特征。\n\n"
    "任务要求：\n"
    "- 在输出最终打分前，必须先在 thought_process 中逐步分析事实依据与逻辑推导，"
    "严禁先下结论再凑分数；\n"
    "- 优先搜索权威新闻媒体、机构、学术论文等可靠来源作为证据；"
    "- 必须输出三个独立维度评分（0-100 整数）：fact_score（事实一致性）、"
    "logic_score（逻辑严谨性）、source_score（信源可靠度），整体 score 与三者保持一致；\n"
    "- 将内容拆解为 2-3 个核心主张（claims），对每个主张单独打分和判定；\n"
    "- 给出 2-5 句 reasoning 分析；\n"
    "- 评估 bias_warning：若存在标题党、情绪煽动、虚假数据、断章取义等误导手法，"
    "用一句话指出；若无风险则输出空字符串；\n"
    "- 输出 logic_graph 逻辑拓扑图：nodes 数组（每项含 id、label、group，"
    "group 只能取：主张 / 证据 / 冲突点），edges 数组（每项含 source、target、label，"
    "label 如：支撑 / 反驳），source 与 target 必须对应 nodes 中已有的 id。\n\n"
    "注意：用户消息中标注为“最近/近日/今年”的事件，请以今天的日期为参照判断其时效性，"
    "不要因为不了解最新时事而误判。\n"
    "如果用户消息中包含【联网检索证据】，必须以该证据为主要判断依据；"
    "若证据不足、来源不明或相互矛盾，请如实说明并给出保守评分。\n\n"
    "重要：无论输入内容使用什么语言，你的所有输出字段"
    "（thought_process、reasoning、claims、bias_warning、logic_graph 中的标签）"
    "都必须始终使用中文。\n\n"
    "你必须严格按照如下 JSON 格式输出，不要输出任何多余文字：\n"
    '{"thought_process": "<输出打分前，先在此逐步分析事实依据、证据支撑与逻辑推导>", '
    '"score": <0-100 的整数>, "fact_score": <0-100 的整数>, '
    '"logic_score": <0-100 的整数>, "source_score": <0-100 的整数>, '
    '"reasoning": "<2-5 句中文分析>", '
    '"claims": [{"claim": "<核心主张内容>", "verdict": "<属实|夸大|虚假 三选一>", "score": <0-100 的整数>}], '
    '"bias_warning": "<误导手法风险提示，无风险则为空字符串>", '
    '"logic_graph": {"nodes": [{"id": 1, "label": "<节点标签>", "group": "<主张|证据|冲突点>"}], '
    '"edges": [{"source": 1, "target": 2, "label": "<支撑|反驳>"}]}}'
)

SYSTEM_PROMPT_EN = (
    "Today's date is {{today}} ({{weekday}}). "
    "You are a rigorous professional fact-checker. Judge the truthfulness of the "
    "news content provided by the user from the following four dimensions:\n"
    "1. Factual consistency: whether the content matches established facts and common sense;\n"
    "2. Logical soundness: whether the argument contains leaps or self-contradictions;\n"
    "3. Source credibility: whether citations, data and attributions are suspicious;\n"
    "4. Misleading techniques: exaggerated headlines, quoting out of context, emotional "
    "manipulation, fabricated data, etc.\n\n"
    "Task requirements:\n"
    "- Before outputting the final scores, you MUST first analyze the factual evidence "
    "and logical deduction step by step in the thought_process field; it is strictly "
    "forbidden to reach a conclusion first and then fit the scores to it;\n"
    "- Prioritize searching for evidence from authoritative news outlets, institutions, and academic papers;\n"
    "- You must output three independent dimension scores (integers 0-100): "
    "fact_score (factual consistency), logic_score (logical soundness), "
    "source_score (source credibility); the overall score must be consistent with them;\n"
    "- Break the content into 2-3 core claims, scoring and judging each claim separately;\n"
    "- Provide a 2-5 sentence reasoning;\n"
    "- Evaluate bias_warning: if clickbait, emotional manipulation, fabricated data or "
    "quoting out of context is present, point it out in one sentence; output an empty "
    "string if there is no risk;\n"
    "- Output a logic_graph: a nodes array (each item with id, label, group; group must be "
    "one of: claim / evidence / conflict), and an edges array (each item with source, "
    "target, label such as supports / refutes); source and target must reference existing "
    "node ids.\n\n"
    "Note: for events described as \"recently / this year\", judge their timeliness against "
    "today's date; do not misjudge recent events as future events just because they are "
    "outside your training data.\n"
    "If the user message contains [Web Search Evidence], you must use it as the primary "
    "basis for your judgment; if the evidence is insufficient, of unknown origin, or "
    "contradictory, state so honestly and give a conservative score.\n\n"
    "IMPORTANT: Regardless of the language of the input content, all of your output fields "
    "(thought_process, reasoning, claims, bias_warning, and labels inside logic_graph) "
    "must ALWAYS be in English.\n\n"
    "You must output strictly in the following JSON format with no extra text:\n"
    '{"thought_process": "<reason step by step about the factual evidence and logic here BEFORE giving the final scores>", '
    '"score": <integer 0-100>, "fact_score": <integer 0-100>, '
    '"logic_score": <integer 0-100>, "source_score": <integer 0-100>, '
    '"reasoning": "<2-5 sentence analysis>", '
    '"claims": [{"claim": "<core claim>", "verdict": "<true|exaggerated|false>", "score": <integer 0-100>}], '
    '"bias_warning": "<misleading-technique risk note, empty string if none>", '
    '"logic_graph": {"nodes": [{"id": 1, "label": "<node label>", "group": "<claim|evidence|conflict>"}], '
    '"edges": [{"source": 1, "target": 2, "label": "<supports|refutes>"}]}}'
)


def get_system_prompt(lang: str) -> str:
    """Dynamically build the system prompt: inject today's date to prevent the
    model from misjudging recent events as "future events"."""
    today = datetime.date.today()
    if lang == "zh":
        # Note: strftime on Windows cannot encode Chinese format strings, so use string concatenation
        date_str = f"{today.year}年{today.month}月{today.day}日"
        weekday = WEEKDAYS_ZH[today.weekday()]
        template = SYSTEM_PROMPT_ZH
    else:
        date_str = f"{MONTHS_EN[today.month - 1]} {today.day}, {today.year}"
        weekday = WEEKDAYS_EN[today.weekday()]
        template = SYSTEM_PROMPT_EN
    return template.replace("{{today}}", date_str).replace("{{weekday}}", weekday)


# ============================ Data fetching module ============================
def is_url(text: str) -> bool:
    """Determine whether the user input is a URL."""
    return bool(URL_PATTERN.match(text.strip()))


async def fetch_webpage(url: str) -> str:
    """
    Asynchronously fetch the webpage HTML.
    Carries browser UA headers, auto-follows redirects, with timeout control
    and exception handling.
    """
    try:
        async with httpx.AsyncClient(
            headers=BROWSER_HEADERS,
            follow_redirects=True,
            timeout=FETCH_TIMEOUT,
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except httpx.TimeoutException as exc:
        raise RuntimeError("网页抓取超时，请稍后重试") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"网页返回异常状态码 {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"网页请求失败：{exc}") from exc


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_webpage_cached(url: str) -> str:
    """
    Cached webpage fetching entry (synchronous wrapper).
    - Re-checking the same URL within ttl=3600 seconds hits the cache directly,
      avoiding unnecessary network latency;
    - Fetching failures raise exceptions (Streamlit does not cache exceptions),
      so retries re-fetch.
    """
    return asyncio.run(fetch_webpage(url))


def extract_article_text(html: str, lang: str = "zh") -> str:
    """
    Extract the news body from HTML:
    1. Remove noisy tags such as script/style/nav/footer;
    2. Extract the meta summary (og:description/description), so content is still
       available when paywalled pages hide the body;
    3. Prefer <article>/<main>/body container class, falling back to <body>;
    4. Keep only paragraphs >= 20 characters to avoid navigation residue; fall back
       to the full text if the body is too short;
    5. Prepend the title to give the model context.
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as exc:
        raise RuntimeError(f"HTML 解析失败：{exc}") from exc

    for tag in soup(["script", "style", "noscript", "iframe", "nav",
                     "footer", "header", "aside", "form", "button"]):
        tag.decompose()

    # meta summary: key fallback when the body of paywalled/JS-rendered pages is invisible
    meta_desc = ""
    for prop in ("og:description", "twitter:description", "description"):
        meta = (soup.find("meta", attrs={"property": prop})
                or soup.find("meta", attrs={"name": prop}))
        if meta and meta.get("content"):
            meta_desc = str(meta["content"]).strip()
            break

    # Body container localization: prefer semantic tags, then heuristically pick
    # the longest candidate by class keywords
    node = soup.find("article") or soup.find("main")
    if node is None:
        best_node, best_len = None, 0
        for div in soup.find_all(["div", "section"]):
            cls = " ".join(div.get("class") or [])
            if any(k in cls for k in ("article-content", "entry-content",
                                      "post-content", "story-content",
                                      "article-body", "news-content")):
                length = len(div.get_text(" ", strip=True))
                if length > best_len:
                    best_node, best_len = div, length
        node = best_node
    if node is None:
        node = soup.body or soup

    paragraphs = [p.get_text(" ", strip=True) for p in node.find_all("p")]
    paragraphs = [p for p in paragraphs if len(p) >= 20]
    body_text = "\n".join(paragraphs).strip()
    if len(body_text) < 50:
        body_text = node.get_text("\n", strip=True)

    parts: List[str] = []
    title = soup.title.get_text(strip=True) if soup.title else ""
    if title:
        prefix = "标题：" if lang == "zh" else "Title: "
        parts.append(f"{prefix}{title}")
    if meta_desc:
        desc_prefix = "摘要：" if lang == "zh" else "Summary: "
        parts.append(f"{desc_prefix}{meta_desc}")
    if body_text:
        parts.append(body_text)
    return "\n\n".join(parts)


def build_user_message(content: str, evidence: str = "", lang: str = "zh") -> str:
    """Assemble the user message: optionally inject web-search evidence; truncate
    overlong bodies; bilingual template."""
    if len(content) > MAX_CONTENT_CHARS:
        cutoff = "\n……（内容过长，已截断）" if lang == "zh" else "\n... (content truncated)"
        content = content[:MAX_CONTENT_CHARS] + cutoff
    if lang == "zh":
        message = f"请核查以下内容的真实性：\n\n{content}"
        if evidence:
            message = f"{message}\n\n{evidence}"
        message += "\n\n请务必只输出一个 JSON 对象，不要包含任何其他文字或 Markdown 格式。"
    else:
        message = f"Please fact-check the truthfulness of the following content:\n\n{content}"
        if evidence:
            message = f"{message}\n\n{evidence}"
        message += ("\n\nYou must output only a single JSON object, "
                    "with no other text or Markdown formatting.")
    return message


# ============================ Web search module ============================
# Filter out domains without a body / video domains, so they do not enter the
# prompt as invalid evidence
EXCLUDED_DOMAINS = ("youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "instagram.com")


class EvidenceNotFoundError(Exception):
    """Raised when the web search returns no results; used to bypass the
    Streamlit cache (exceptions are not cached)."""


def _search_web_sync(query: str, max_results: int) -> List[Dict[str, str]]:
    """Run the DuckDuckGo search synchronously (ddgs is a sync library); return
    an empty list on failure."""
    if not HAS_SEARCH:
        return []
    try:
        raw = DDGS().text(query, max_results=max_results)
    except Exception:
        return []
    items = []
    for item in raw or []:
        url = str(item.get("href") or item.get("url") or "")
        if any(domain in url for domain in EXCLUDED_DOMAINS):
            continue
        items.append({
            "title": str(item.get("title") or ""),
            "url": url,
            "body": str(item.get("body") or ""),
        })
    return items


async def search_web(query: str,
                     max_results: int = SEARCH_MAX_RESULTS) -> List[Dict[str, str]]:
    """Async wrapper: run the DuckDuckGo search in a separate thread to avoid
    blocking the event loop."""
    return await asyncio.to_thread(_search_web_sync, query, max_results)


async def fetch_evidence_pages(items: List[Dict[str, str]],
                               limit: int = SEARCH_PAGE_FETCH_N,
                               lang: str = "zh") -> List[str]:
    """Concurrently fetch body excerpts of the top limit search results; silently
    skip individual failures."""
    async def grab(item: Dict[str, str]) -> str:
        url = item.get("url", "")
        if not url:
            return ""
        try:
            async with httpx.AsyncClient(
                headers=BROWSER_HEADERS, follow_redirects=True,
                timeout=SEARCH_PAGE_TIMEOUT,
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return extract_article_text(resp.text, lang)[:SEARCH_SNIPPET_CHARS]
        except Exception:
            return ""

    gathered = await asyncio.gather(
        *[grab(it) for it in items[:limit]], return_exceptions=True
    )
    return [r if isinstance(r, str) else "" for r in gathered]


async def collect_evidence(query: str, lang: str = "zh") -> str:
    """Search the web for evidence and assemble it into a text block for the
    model (bilingual); return an empty string when there are no results."""
    items = await search_web(query)
    if not items:
        return ""
    pages = await fetch_evidence_pages(items, lang=lang)

    if lang == "zh":
        lines = ["【联网检索证据】（来自搜索引擎，供事实核查参考）", ""]
        labels = ("来源", "标题", "链接", "摘要", "正文节选")
    else:
        lines = ["[Web Search Evidence] (retrieved from a search engine for fact-checking reference)", ""]
        labels = ("Source", "Title", "URL", "Snippet", "Excerpt")

    for idx, item in enumerate(items):
        lines.append(f"{labels[0]} {idx + 1}")
        lines.append(f"{labels[1]}：{item['title']}")
        if item["url"]:
            lines.append(f"{labels[2]}：{item['url']}")
        if item["body"]:
            lines.append(f"{labels[3]}：{item['body']}")
        if idx < len(pages) and pages[idx]:
            lines.append(f"{labels[4]}：{pages[idx]}")
        lines.append("")
    return "\n".join(lines)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_evidence_cached(query: str, lang: str) -> str:
    """
    Cached web-search entry (synchronous wrapper).
    - Re-searching the same keyword within ttl=3600 seconds hits the cache directly;
    - Raises EvidenceNotFoundError (not cached) when there are no results,
      avoiding caching a "rate-limit-induced empty result" for an hour.
    """
    evidence = asyncio.run(collect_evidence(query, lang))
    if not evidence:
        raise EvidenceNotFoundError(query)
    return evidence


def _extract_title(content: str, lang: str = "zh") -> str:
    """Extract the title line from the fetched body, used as the web-search
    keyword."""
    prefix = "标题：" if lang == "zh" else "Title: "
    for line in content.splitlines():
        if line.startswith(prefix):
            return line.replace(prefix, "", 1).strip()
    return ""


# ============================ API call module ============================
def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract the JSON object from the model output:
    1. Try a full json.loads parse;
    2. Strip markdown code fences (```json ... ```) and parse again;
    3. Extract the first complete JSON object with bracket-depth matching
       (skipping string content);
    4. On parse failure, repair trailing commas and retry.
    """
    if not text:
        return None
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    data = json.loads(candidate)
                    return data if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    # Repair a common flaw: drop trailing commas and retry
                    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        data = json.loads(repaired)
                        return data if isinstance(data, dict) else None
                    except json.JSONDecodeError:
                        return None
    return None


def extract_score_reasoning(text: str) -> Optional[Dict[str, Any]]:
    """
    Fallback extraction: when the model's JSON cannot be fully parsed, use regex
    to extract the score / fact_score / logic_score / source_score /
    thought_process / reasoning fields directly, rescuing the call as much as
    possible.
    """
    if not isinstance(text, str):
        return None
    score_m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if not score_m:
        return None
    reasoning_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    reasoning = ""
    if reasoning_m:
        reasoning = reasoning_m.group(1).replace('\\"', '"').replace("\\n", "\n")
    thought_m = re.search(r'"thought_process"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
    out: Dict[str, Any] = {"score": float(score_m.group(1)), "reasoning": reasoning}
    if thought_m:
        out["thought_process"] = thought_m.group(1).replace('\\"', '"').replace("\\n", "\n")
    for key in DIM_KEYS:
        m = re.search(r'"' + key + r'"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_claims(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Safely parse the claims array (backward compatible):
    - Returns an empty list when the model did not return one / returned a
      non-list, never raising;
    - Validates field types and score ranges per item, dropping invalid entries.
    """
    claims: List[Dict[str, Any]] = []
    raw_claims = parsed.get("claims", [])
    if not isinstance(raw_claims, list):
        return claims
    for item in raw_claims[:10]:
        if not isinstance(item, dict):
            continue
        try:
            cscore = int(round(float(item.get("score"))))
        except (TypeError, ValueError):
            continue
        claims.append({
            "claim": str(item.get("claim", "")).strip(),
            "verdict": str(item.get("verdict", "")).strip(),
            "score": max(0, min(100, cscore)),
        })
    return claims


def parse_logic_graph(parsed: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Safely parse the logic_graph topology structure (fault tolerant):
    - Returns None for invalid structures / missing nodes / dangling edges,
      letting the front end gracefully degrade to plain text;
    - All ids are normalized to strings, labels are truncated, and isolated
      edges are dropped; never raises.
    """
    graph = parsed.get("logic_graph")
    if not isinstance(graph, dict):
        return None
    raw_nodes = graph.get("nodes")
    raw_edges = graph.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        return None

    nodes: List[Dict[str, str]] = []
    node_ids = set()
    for n in raw_nodes[:30]:
        if not isinstance(n, dict) or n.get("id") is None:
            continue
        nid = str(n["id"])
        if nid in node_ids:
            continue
        node_ids.add(nid)
        nodes.append({
            "id": nid,
            "label": str(n.get("label") or nid)[:80],
            "group": str(n.get("group") or "").strip()[:20],
        })
    if not nodes:
        return None

    edges: List[Dict[str, str]] = []
    for e in raw_edges[:60]:
        if not isinstance(e, dict):
            continue
        src, dst = e.get("source"), e.get("target")
        if src is None or dst is None:
            continue
        src_s, dst_s = str(src), str(dst)
        # Drop dangling edges (source/target must reference existing nodes)
        if src_s not in node_ids or dst_s not in node_ids:
            continue
        edges.append({
            "source": src_s,
            "target": dst_s,
            "label": str(e.get("label") or "")[:40],
        })
    return {"nodes": nodes, "edges": edges}


def _clamp_score(value: Any, fallback: float) -> int:
    """Safely convert any value to an integer in 0-100, falling back when the
    conversion fails."""
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return max(0, min(100, int(round(float(fallback)))))


async def call_model(client: httpx.AsyncClient, model: str,
                     content: str, lang: str = "zh",
                     system_prompt: Optional[str] = None,
                     timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Call a single base model for fact-checking.
    - Prefers response_format=json_object to force JSON output;
    - If the model does not support it (returns 400), drops the field and retries once;
    - When system_prompt is empty, uses the default fact-checking prompt; special
      tasks such as Meta-Judge can pass a custom prompt;
    - When timeout is empty, uses the global REQUEST_TIMEOUT; a halved timeout can
      be passed during degraded retries;
    - Extracts request_id, 3-D scores (fact/logic/source), thought_process,
      reasoning, claims, bias_warning, logic_graph, and validates their legality;
    - Any missing field degrades safely (backward compatible), never raising.
    """
    effective_timeout = timeout if timeout is not None else REQUEST_TIMEOUT
    result: Dict[str, Any] = {
        "model": model, "ok": False, "score": None, "reasoning": None,
        "thought_process": None,
        "fact_score": None, "logic_score": None, "source_score": None,
        "claims": [], "bias_warning": "", "logic_graph": None,
        "request_id": None, "raw": None, "error": None,
    }
    endpoint = f"{API_BASE_URL}/chat/completions"
    messages = [
        {"role": "system", "content": system_prompt or get_system_prompt(lang)},
        {"role": "user", "content": content},
    ]

    response = None
    for use_json_mode in (True, False):
        payload: Dict[str, Any] = {
            "model": model, "messages": messages, "temperature": 0.1,
            "max_tokens": MAX_TOKENS,
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await client.post(endpoint, json=payload,
                                          timeout=effective_timeout)
        except httpx.TimeoutException:
            result["error"] = t(lang, "err_timeout").format(t=effective_timeout)
            return result
        except httpx.HTTPError as exc:
            result["error"] = t(lang, "err_network").format(e=exc)
            return result
        if use_json_mode and response.status_code == 400:
            continue  # Model does not support response_format; drop it and retry once
        break

    # Extract Request ID: the Gonka gateway's tracking ID is in the x-request-id
    # response header (highest priority)
    data: Dict[str, Any] = {}
    try:
        data = response.json()
    except Exception:
        data = {}
    result["request_id"] = (response.headers.get("x-request-id")
                            or data.get("request_id") or data.get("id"))

    if response.status_code != 200:
        err = data.get("error", {})
        msg = err.get("message") if isinstance(err, dict) else str(err)
        result["error"] = t(lang, "err_http").format(
            code=response.status_code, msg=msg or response.text[:200])
        return result

    try:
        answer = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        result["error"] = t(lang, "err_structure").format(e=exc)
        return result
    if not isinstance(answer, str) or not answer.strip():
        result["error"] = t(lang, "err_structure").format(e="空回复内容")
        return result

    result["raw"] = answer
    parsed = extract_json_block(answer)
    if parsed is None:
        parsed = extract_score_reasoning(answer)  # Fallback regex extraction to rescue as much as possible
    if parsed is None:
        result["error"] = t(lang, "err_json")
        return result

    try:
        # Fall back to final_score when score is missing, compatible with tasks
        # such as Meta-Judge
        score = int(round(float(parsed.get("score", parsed.get("final_score")))))
    except (TypeError, ValueError):
        result["error"] = t(lang, "err_score")
        return result
    score = max(0, min(100, score))

    # Three dimension scores: fall back to the overall score when missing
    # (backward compatible), so matrix M can always be built
    result["score"] = score
    result["fact_score"] = _clamp_score(parsed.get("fact_score"), score)
    result["logic_score"] = _clamp_score(parsed.get("logic_score"), score)
    result["source_score"] = _clamp_score(parsed.get("source_score"), score)
    result["thought_process"] = str(parsed.get("thought_process", "")).strip()
    result["reasoning"] = str(parsed.get("reasoning", "")).strip()
    result["claims"] = parse_claims(parsed)
    result["bias_warning"] = str(parsed.get("bias_warning", "") or "").strip()
    result["logic_graph"] = parse_logic_graph(parsed)  # Invalid structures return None; the front end degrades
    result["ok"] = True
    return result


async def run_all_models(content: str, models: List[str],
                         api_key: str, lang: str = "zh") -> List[Dict[str, Any]]:
    """
    Two-phase request to all base models (degraded retry strategy for unstable APIs):
    1. Round 1: asyncio.gather concurrent requests; a single model's
       timeout/exception is isolated and does not affect the others' results;
    2. Round 2: filter the models with ok == False (failed or timed out) from
       round 1, retry them sequentially in serial order (avoiding re-triggering
       the provider's concurrency limit), with the timeout halved (REQUEST_TIMEOUT / 2);
    3. Successful round-2 retries are backfilled into the final results list,
       which is then handed to the matrix consensus engine.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def failed_result(model: str, error: Any) -> Dict[str, Any]:
        """Build a unified failure placeholder result; fields stay consistent
        with a successful result."""
        return {
            "model": model, "ok": False, "score": None, "reasoning": None,
            "thought_process": None,
            "fact_score": None, "logic_score": None, "source_score": None,
            "claims": [], "bias_warning": "", "logic_graph": None,
            "request_id": None, "raw": None,
            "error": t(lang, "err_unhandled").format(e=error),
        }

    async with httpx.AsyncClient(headers=headers) as client:
        # Round 1: request all models concurrently with asyncio.gather
        tasks = [call_model(client, model, content, lang) for model in models]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Dict[str, Any]] = []
        for model, res in zip(models, gathered):
            if isinstance(res, Exception):
                results.append(failed_result(model, res))
            else:
                results.append(res)

        # Round 2: serially retry the round-1 failed/timed-out models with a
        # halved timeout to avoid concurrency limits
        retry_indices = [i for i, r in enumerate(results) if not r.get("ok")]
        for i in retry_indices:
            retry = await call_model(client, results[i]["model"], content, lang,
                                     timeout=REQUEST_TIMEOUT / 2)
            if not isinstance(retry, Exception):
                results[i] = retry  # Retry succeeded: backfill and update the final result
    return results


# ====================== Meta-Judge (Arbiter) Module ======================
META_JUDGE_PROMPT_ZH = (
    "你是一名公正、权威的仲裁者（Meta-Judge）。多位底座大模型对同一条内容"
    "给出了存在重大分歧的三维评分与思维链。请仔细阅读各模型的思维链"
    "（thought_process）、推理与评分，综合各方观点，指出核心分歧点，"
    "并给出公正的最终定论。所有输出必须使用中文。\n"
    "你必须严格输出如下 JSON 格式，不要输出任何多余文字：\n"
    '{"summary": "<总结各方观点与核心分歧>", '
    '"final_verdict": "<最终定论：可信 / 存疑 / 虚假，并说明理由>", '
    '"final_score": <0-100 的整数>}'
)

META_JUDGE_PROMPT_EN = (
    "You are a fair and authoritative Meta-Judge. Multiple base models returned "
    "significantly divergent 3-dimensional scores and chains of thought for the same "
    "content. Read each model's thought_process, reasoning and scores carefully, "
    "synthesize the viewpoints, point out the core disagreements, and give a fair "
    "final ruling. All output must be in English.\n"
    "You must output strictly in the following JSON format with no extra text:\n"
    '{"summary": "<synthesize each side and the core disagreements>", '
    '"final_verdict": "<final ruling: credible / questionable / false, with reasons>", '
    '"final_score": <integer 0-100>}'
)


def get_meta_judge_prompt(lang: str) -> str:
    """Return the Meta-Judge system prompt (no date placeholder; return it directly)."""
    return META_JUDGE_PROMPT_ZH if lang == "zh" else META_JUDGE_PROMPT_EN


def build_meta_judge_message(ok_results: List[Dict[str, Any]], content: str,
                             evidence: str, lang: str = "zh") -> str:
    """Concatenate each successful model's chain of thought and scores into the
    Meta-Judge user message, carrying the current web-search evidence."""
    if lang == "zh":
        lines = [
            "多位模型对以下内容给出了重大分歧的评分，请你仲裁。",
            "",
            "【待核查内容】",
            content[:MAX_CONTENT_CHARS],
            "",
        ]
        if evidence:
            lines += ["【联网检索证据】", evidence, ""]
        lines.append("【各模型观点】")
        for idx, r in enumerate(ok_results, 1):
            lines.append(f"{idx}. {r['model']}")
            lines.append(
                f"   综合评分：{r['score']}/100；事实一致性：{r['fact_score']}；"
                f"逻辑严谨性：{r['logic_score']}；信源可靠度：{r['source_score']}"
            )
            tp = str(r.get("thought_process") or "").strip()
            lines.append(f"   思维链：{tp or t(lang, 'meta_judge_no_thought')}")
            if r.get("reasoning"):
                lines.append(f"   推理分析：{r['reasoning']}")
    else:
        lines = [
            "Multiple models returned significantly divergent scores for the "
            "content below. Please arbitrate.",
            "",
            "[Content to Fact-Check]",
            content[:MAX_CONTENT_CHARS],
            "",
        ]
        if evidence:
            lines += ["[Web Search Evidence]", evidence, ""]
        lines.append("[Viewpoints of Each Model]")
        for idx, r in enumerate(ok_results, 1):
            lines.append(f"{idx}. {r['model']}")
            lines.append(
                f"   Overall score: {r['score']}/100; fact: {r['fact_score']}; "
                f"logic: {r['logic_score']}; source: {r['source_score']}"
            )
            tp = str(r.get("thought_process") or "").strip()
            lines.append(f"   Chain of thought: {tp or t(lang, 'meta_judge_no_thought')}")
            if r.get("reasoning"):
                lines.append(f"   Reasoning: {r['reasoning']}")
    return "\n".join(lines)


async def run_meta_judge(results: List[Dict[str, Any]], content: str,
                         evidence: str, api_key: str,
                         lang: str = "zh") -> Optional[Dict[str, Any]]:
    """
    High-disagreement arbitration: when the disagreement between models exceeds
    the threshold, extract the thought_process and scores of all successful
    models, concatenate them into a new prompt, carry the existing web-search
    evidence, and call DEFAULT_MODELS[0] (DeepSeek) as the arbiter to summarize
    each viewpoint and give a final ruling.
    """
    ok_results = [r for r in results if r.get("ok")]
    if not ok_results or not DEFAULT_MODELS:
        return None
    arbiter_model = DEFAULT_MODELS[0]
    user_msg = build_meta_judge_message(ok_results, content, evidence, lang)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        res = await call_model(client, arbiter_model, user_msg, lang,
                               system_prompt=get_meta_judge_prompt(lang),
                               timeout=REQUEST_TIMEOUT)
    if not res.get("ok") or not res.get("raw"):
        return {"ok": False, "model": arbiter_model,
                "error": res.get("error") or t(lang, "unknown_error")}
    parsed = extract_json_block(res["raw"]) or {}
    final_verdict = str(parsed.get("final_verdict", "") or "").strip()
    if not final_verdict:
        final_verdict = t(lang, "verdicts").get(classify_verdict(res["score"]), "")
    return {
        "ok": True,
        "model": arbiter_model,
        "summary": str(parsed.get("summary", "") or res.get("reasoning") or "").strip(),
        "final_verdict": final_verdict,
        "final_score": res["score"],
        "thought_process": res.get("thought_process") or "",
        "request_id": res.get("request_id"),
    }


# ====================== Matrix consensus engine (linear algebra) ======================
def classify_verdict(avg: float) -> str:
    """Return the key of the verdict for the final score (the wording is
    translated per language in the UI layer)."""
    if avg >= 80:
        return "high"
    if avg >= 60:
        return "medium"
    if avg >= 40:
        return "low"
    return "suspect"


def compute_consensus(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Multi-dimensional matrix consensus engine (linear algebra):

    1. Build the N×3 score matrix M: each row is a model's feature vector
       mᵢ = [fact_score, logic_score, source_score];
    2. Compute the centroid (mean vector): c = (1/N) Σ mᵢ;
    3. Compute the Euclidean distance (L2 norm) of each vector to the centroid:
       dᵢ = ‖mᵢ − c‖₂;
    4. Dynamic weights (Gaussian-kernel decay + L1 normalization):
       wᵢ = exp(−dᵢ² / (2σ_d²)), σ_d = std(d) (std of the distance distribution);
       degrades to equal weights when σ_d ≈ 0 (all highly consistent);
       weight vector W sums to 1;
    5. Weighted matrix product: v = WᵀM (1×3 weighted composite vector);
    6. Final Truth Score = mean(v) = (1/3) Σ vⱼ;
    7. Model disagreement d̄ = (1/N) Σ dᵢ (average Euclidean distance).
    """
    valid = [r for r in results
             if r.get("ok") and all(r.get(k) is not None for k in DIM_KEYS)]
    total = len(results)
    if not valid:
        return {"avg_score": None, "disagreement": None, "verdict": "unknown",
                "valid_count": 0, "total_count": total,
                "matrix": None, "weights": None, "centroid": None,
                "weighted_vector": None, "model_order": []}

    # 1. N×3 score matrix
    M = np.array([[r["fact_score"], r["logic_score"], r["source_score"]]
                  for r in valid], dtype=np.float64)
    # 2. Centroid (mean vector)
    centroid = M.mean(axis=0)
    # 3. Euclidean distances (L2 norm)
    distances = np.linalg.norm(M - centroid, axis=1)
    # 4. Gaussian-kernel dynamic weights
    if len(distances) > 1 and float(distances.std()) > 1e-6:
        sigma_d = float(distances.std())
        raw_w = np.exp(-(distances ** 2) / (2.0 * sigma_d ** 2))
    else:
        # Degrade to equal weights when all models are highly consistent
        # (or there is only one model)
        raw_w = np.ones(len(distances), dtype=np.float64)
    w_sum = float(raw_w.sum())
    weights = raw_w / w_sum if w_sum > 1e-12 else np.ones(len(distances)) / len(distances)
    # 5. Weighted matrix product: v = WᵀM
    weighted_vector = weights @ M
    # 6. Final Truth Score
    truth_score = round(float(weighted_vector.mean()), 1)
    # 7. Model disagreement (average Euclidean distance)
    disagreement = round(float(distances.mean()), 1)

    return {
        "avg_score": truth_score,
        "disagreement": disagreement,
        "verdict": classify_verdict(truth_score),
        "valid_count": len(valid), "total_count": total,
        "matrix": M, "weights": weights, "centroid": centroid,
        "weighted_vector": weighted_vector,
        "model_order": [r["model"] for r in valid],
    }


# ============================ Report generation module ============================
def build_report(consensus: Dict[str, Any], results: List[Dict[str, Any]],
                 content: str, lang: str,
                 ts: Optional[datetime.datetime] = None) -> str:
    """Assemble the complete Markdown fact-check report: time, scores, matrix,
    claims, chains of thought + reasoning traces, Request IDs."""
    ts = ts or datetime.datetime.now()
    lines: List[str] = [f"# {t(lang, 'report_title')}", ""]

    lines.append(f"- {t(lang, 'report_time')}：{ts.strftime('%Y-%m-%d %H:%M:%S')}")
    if consensus["avg_score"] is None:
        lines.append(f"- {t(lang, 'report_score')}：{t(lang, 'verdicts')['unknown']}")
    else:
        lines.append(f"- {t(lang, 'report_score')}：{consensus['avg_score']:.1f} / 100")
        lines.append(f"- {t(lang, 'report_verdict')}：{t(lang, 'verdicts')[consensus['verdict']]}")
        lines.append(f"- {t(lang, 'report_disagree')}：{consensus['disagreement']:.1f}")
    lines.append("")

    lines.append(f"## {t(lang, 'report_input')}")
    lines.append("")
    lines.append(content[:500].strip() or "（无）")
    lines.append("")

    # Score matrix and weights
    if consensus.get("matrix") is not None:
        lines.append(f"## {t(lang, 'report_matrix')}")
        lines.append("")
        header = ["模型", t(lang, "col_fact"), t(lang, "col_logic"),
                  t(lang, "col_source"), t(lang, "col_weight")]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for i, model in enumerate(consensus["model_order"]):
            row = [model,
                   f"{consensus['matrix'][i, 0]:.0f}",
                   f"{consensus['matrix'][i, 1]:.0f}",
                   f"{consensus['matrix'][i, 2]:.0f}",
                   f"{consensus['weights'][i]:.3f}"]
            lines.append("| " + " | ".join(row) + " |")
        vec = consensus["weighted_vector"]
        lines.append("")
        lines.append(f"{t(lang, 'report_vector')} v = WᵀM = "
                     f"({vec[0]:.1f}, {vec[1]:.1f}, {vec[2]:.1f})；"
                     f"{t(lang, 'report_score')} = mean(v) = {consensus['avg_score']:.1f} / 100")
        lines.append("")

    lines.append(f"## {t(lang, 'report_claims')}")
    lines.append("")
    for res in results:
        if not res.get("ok"):
            continue
        lines.append(f"### {res['model']}（{res['score']}/100 · "
                     f"{t(lang, 'col_fact')} {res['fact_score']} / "
                     f"{t(lang, 'col_logic')} {res['logic_score']} / "
                     f"{t(lang, 'col_source')} {res['source_score']}）")
        lines.append("")
        if res.get("claims"):
            for c in res["claims"]:
                lines.append(f"- [{c['verdict'] or '-'}] {c['claim'] or '-'}（{c['score']}/100）")
        else:
            lines.append(t(lang, "report_no_claims"))
        lines.append("")

    lines.append(f"## {t(lang, 'report_trace')}")
    lines.append("")
    for res in results:
        lines.append(f"### {res['model']}")
        lines.append("")
        if res.get("ok"):
            if res.get("thought_process"):
                lines.append(f"**{t(lang, 'thought_label')}**")
                lines.append("")
                lines.append(res["thought_process"])
                lines.append("")
            lines.append(res.get("reasoning") or t(lang, "no_reasoning"))
        else:
            lines.append(f"*{t(lang, 'error_label').format(err=res.get('error') or t(lang, 'unknown_error'))}*")
        lines.append("")

    lines.append(f"## {t(lang, 'report_bias')}")
    lines.append("")
    bias_lines = [f"- **{res['model']}**：{res['bias_warning']}"
                  for res in results if res.get("ok") and res.get("bias_warning")]
    lines.extend(bias_lines or [t(lang, "report_no_bias")])
    lines.append("")

    lines.append(f"## {t(lang, 'report_reqids')}")
    lines.append("")
    lines.append("```json")
    req_ids = {res["model"]: res.get("request_id") or t(lang, "not_obtained")
               for res in results}
    lines.append(json.dumps(req_ids, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append(f"*{t(lang, 'report_footer')}*")
    return "\n".join(lines)


# ============================ UI rendering module ============================
def render_logic_graph(graph: Dict[str, Any]) -> None:
    """
    Render a draggable logic topology graph with pyvis.
    - cdn_resources="in_line": vis.js is inlined directly into the HTML, with no
      external CDN dependency;
    - Node colors are distinguished by group (claim=blue / evidence=green / conflict=red);
    - The physics engine is on by default, so users can drag nodes;
    - Also removes the external Bootstrap CSS referenced by the template
      (decorative only, does not affect rendering).
    """
    if not HAS_GRAPH_LIB:
        return
    net = Network(height="420px", width="100%", directed=True,
                  notebook=False, cdn_resources="in_line")
    for n in graph["nodes"]:
        color = GRAPH_GROUP_COLORS.get(n["group"], GRAPH_DEFAULT_COLOR)
        net.add_node(n["id"], label=n["label"], color=color, shape="box",
                     font={"size": 16, "face": "sans-serif"})
    for e in graph["edges"]:
        net.add_edge(e["source"], e["target"], label=e["label"], color="#AAAAAA")
    html = net.generate_html()
    # Remove external Bootstrap CSS/JS references (decorative only, does not
    # affect rendering), guaranteeing fully offline use
    html = re.sub(r'<script[^>]*cdn\.jsdelivr\.net[^>]*>\s*</script>', "", html)
    html = re.sub(r'<link[^>]*cdn\.jsdelivr\.net[^>]*>', "", html)
    st.components.v1.html(html, height=470, scrolling=False)


def render_results(consensus: Dict[str, Any],
                   results: List[Dict[str, Any]], lang: str) -> None:
    """Render the check results: high-disagreement warning and arbitration,
    evidence, Truth Score, matrix M and weights W, topology graph, credentials,
    and report download."""
    avg = consensus["avg_score"]
    disagreement = consensus.get("disagreement")
    meta_judge = st.session_state.get("last_meta_judge")

    # High-disagreement warning + Meta-Judge ruling (prominently shown at the
    # top of the results area)
    if (avg is not None and disagreement is not None
            and disagreement > DISAGREEMENT_THRESHOLD):
        st.error(t(lang, "disagree_warning").format(
            d=disagreement, thr=DISAGREEMENT_THRESHOLD))
        if meta_judge and meta_judge.get("ok"):
            with st.container(border=True):
                st.markdown(f"**{t(lang, 'meta_judge_title')}**")
                st.caption(f"{t(lang, 'meta_judge_model_label')}："
                           f"{meta_judge['model']}")
                st.markdown(f"**{t(lang, 'meta_judge_final')}**\n\n"
                            f"{meta_judge['final_verdict']}")
                st.metric(t(lang, "meta_judge_score_label"),
                          f"{meta_judge['final_score']} / 100")
                if meta_judge.get("summary"):
                    st.markdown(f"**{t(lang, 'meta_judge_summary')}**\n\n"
                                f"{meta_judge['summary']}")
                if meta_judge.get("thought_process"):
                    with st.expander(t(lang, "thought_label")):
                        st.text(meta_judge["thought_process"])
        elif meta_judge:
            st.warning(t(lang, "meta_judge_failed"))
        else:
            st.info(t(lang, "meta_judge_failed"))

    evidence = st.session_state.get("last_evidence")
    if evidence:
        with st.expander(t(lang, "evidence_expander"), expanded=False):
            st.text(evidence)

    if avg is None:
        st.error(t(lang, "all_failed"))
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(t(lang, "metric_score"), f"{avg:.1f} / 100")
        with c2:
            st.metric(t(lang, "metric_verdict"),
                      t(lang, "verdicts")[consensus["verdict"]])
        with c3:
            st.metric(t(lang, "metric_disagree"), f"{consensus['disagreement']:.1f}")
        st.progress(min(max(avg, 0) / 100.0, 1.0))
        st.caption(t(lang, "consensus_caption").format(
            valid=consensus["valid_count"], total=consensus["total_count"]))

        # Multi-model score comparison bar chart (three dimensions side by side)
        st.markdown(f"**{t(lang, 'chart_title')}**")
        
        # Build a DataFrame with three data columns; Streamlit renders it as a
        # side-by-side bar chart
        chart_df = pd.DataFrame(
            {
                t(lang, "col_fact"): [r["fact_score"] if r.get("ok") and r.get("fact_score") is not None else 0 for r in results],
                t(lang, "col_logic"): [r["logic_score"] if r.get("ok") and r.get("logic_score") is not None else 0 for r in results],
                t(lang, "col_source"): [r["source_score"] if r.get("ok") and r.get("source_score") is not None else 0 for r in results],
            },
            index=[r["model"] for r in results],
        )
        
        st.bar_chart(chart_df, stack=False)
        
        # Optional: if you want the caption below the chart to change too,
        # override it here directly
        if lang == "zh":
            st.caption("纵轴为各模型在三个维度的独立评分。柱高落差直观反映了模型内部的认知分歧。")
        else:
            st.caption("Vertical axis shows the three independent dimension scores. Height differences reveal the model's internal cognitive gap.")

        # N×3 score matrix M and dynamic weights W
        st.markdown(f"**{t(lang, 'matrix_title')}**")
        matrix_df = pd.DataFrame(
            consensus["matrix"],
            index=consensus["model_order"],
            columns=[t(lang, "col_fact"), t(lang, "col_logic"), t(lang, "col_source")],
        )
        matrix_df[t(lang, "col_weight")] = np.round(consensus["weights"], 3)
        st.table(matrix_df)
        st.caption(t(lang, "matrix_caption"))
        vec = consensus["weighted_vector"]
        cent = consensus["centroid"]
        st.caption(t(lang, "weighted_vector_caption").format(
            c=", ".join(f"{x:.1f}" for x in cent),
            v=", ".join(f"{x:.1f}" for x in vec),
            s=f"{avg:.1f}"))

    # Misleading-technique warning: shown only when there is a risk, using a
    # prominent warning style
    for res in results:
        if res.get("ok") and res.get("bias_warning"):
            st.warning(f"**{res['model']}** · {t(lang, 'bias_label')}：{res['bias_warning']}")

    st.divider()
    # Reasoning trace section: upgraded to a draggable logic topology graph;
    # gracefully degrades to plain text when the graph is missing
    st.subheader(t(lang, "reasoning_trace"))
    st.caption(t(lang, "graph_legend"))
    for res in results:
        if res.get("ok"):
            title = res["model"] + t(lang, "score_title").format(score=res["score"])
        else:
            title = res["model"] + t(lang, "failed_title")
        with st.expander(title):
            if res.get("ok"):
                if res.get("logic_graph"):
                    if HAS_GRAPH_LIB:
                        render_logic_graph(res["logic_graph"])
                    else:
                        st.info(t(lang, "graph_missing_lib"))
                else:
                    st.caption(t(lang, "graph_fallback"))
                if res.get("thought_process"):
                    st.markdown(f"**{t(lang, 'thought_label')}**\n\n"
                                f"{res['thought_process']}")
                st.markdown(f"**{t(lang, 'reasoning_label')}**\n\n"
                            f"{res['reasoning'] or t(lang, 'no_reasoning')}")
                if res.get("claims"):
                    st.markdown(f"**{t(lang, 'claims_title')}**")
                    claims_df = pd.DataFrame(res["claims"])
                    claims_df.columns = [t(lang, "col_claim"), t(lang, "col_verdict"),
                                         t(lang, "col_score")]
                    st.table(claims_df)
                if res.get("raw"):
                    with st.expander(t(lang, "raw_output")):
                        st.code(res["raw"], language="json")
            else:
                st.error(t(lang, "error_label").format(
                    err=res.get("error") or t(lang, "unknown_error")))

    st.divider()
    # Decentralized inference verification credential: a visual credibility card
    with st.container(border=True):
        st.markdown(f"**{t(lang, 'credential_title')}**")
        st.caption(t(lang, "credential_network"))
        req_ids = {res["model"]: res.get("request_id") or t(lang, "not_obtained")
                   for res in results}
        st.code(json.dumps(req_ids, ensure_ascii=False, indent=2), language="json")
        st.caption(t(lang, "request_ids_caption"))

    # Fact-check report export
    content = st.session_state.get("last_content", "")
    report_md = build_report(consensus, results, content, lang)
    fname = f"fact_check_report_{datetime.datetime.now():%Y%m%d_%H%M%S}.md"
    st.download_button(
        t(lang, "download_report"),
        data=report_md.encode("utf-8"),
        file_name=fname,
        mime="text/markdown",
        use_container_width=True,
    )


def handle_check(raw_input: str, api_key: str, models: List[str],
                 enable_search: bool, lang: str) -> None:
    """Full business flow after clicking "Start Fact-Check": fetch -> web search
    -> concurrent calls -> matrix consensus -> (Meta-Judge) -> render."""
    text = (raw_input or "").strip()
    if not text:
        st.warning(t(lang, "warn_empty_input"))
        return
    if not api_key.strip():
        st.error(t(lang, "warn_no_api_key"))
        return
    if not models:
        st.warning(t(lang, "warn_no_models"))
        return

    content = text
    search_query = ""
    if is_url(text):
        # URL mode: asynchronously fetch the page with caching and extract the
        # body; on failure prompt the user to paste the text manually
        try:
            with st.spinner(t(lang, "fetching_page")):
                html = fetch_webpage_cached(text)  # No network latency when hitting the cache
                content = extract_article_text(html, lang)
            if len(content.strip()) < 20:
                raise RuntimeError(t(lang, "no_body"))
            search_query = _extract_title(content, lang) or content[:80]
        except Exception as exc:
            st.error(t(lang, "fetch_failed"))
            st.caption(t(lang, "tech_detail").format(exc))
            return
    else:
        # Text mode: use the input content (stripped of trailing punctuation)
        # as the search keyword
        search_query = text[:120].rstrip("？?！!。，,")

    # Web-search evidence: solves the problem that models cannot answer
    # time-sensitive news because they cannot browse the web
    evidence = ""
    if enable_search:
        if not HAS_SEARCH:
            st.warning(t(lang, "no_ddgs"))
        else:
            with st.spinner(t(lang, "searching")):
                try:
                    evidence = get_evidence_cached(search_query, lang)  # With caching
                except EvidenceNotFoundError:
                    evidence = ""
                except Exception:
                    evidence = ""
            if evidence:
                st.caption(t(lang, "evidence_ok"))
            else:
                st.warning(t(lang, "search_no_results"))

    if not is_url(text) and len(content.strip()) < 20 and not evidence:
        st.warning(t(lang, "content_too_short"))

    with st.spinner(t(lang, "calling_models").format(n=len(models))):
        results = asyncio.run(
            run_all_models(build_user_message(content, evidence, lang),
                           models, api_key, lang)
        )
    consensus = compute_consensus(results)

    # High-disagreement warning: launch the Meta-Judge for a second ruling when
    # the disagreement exceeds the threshold
    meta_judge = None
    if (consensus.get("disagreement") is not None
            and consensus["disagreement"] > DISAGREEMENT_THRESHOLD):
        with st.spinner(t(lang, "meta_judge_running")):
            meta_judge = asyncio.run(
                run_meta_judge(results, content, evidence, api_key, lang)
            )

    st.session_state["last_results"] = results
    st.session_state["last_consensus"] = consensus
    st.session_state["last_evidence"] = evidence
    st.session_state["last_content"] = content
    st.session_state["last_meta_judge"] = meta_judge
    st.session_state["last_request_ids"] = {
        res["model"]: res.get("request_id") for res in results
    }
    st.caption(t(lang, "done_caption").format(
        chars=len(content), models=len(models)))
    render_results(consensus, results, lang)


def render_ui() -> None:
    """Streamlit main UI (supports Chinese / English switching)."""
    st.set_page_config(page_title="AI Fact Checker / AI 事实核查器", layout="wide")

    # ---------------- Sidebar: language switching, configuration, and Request IDs ----------------
    with st.sidebar:
        lang = st.segmented_control(
            "界面语言 / Language",
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            default="zh",
        )
        st.divider()
        st.header(t(lang, "sidebar_config"))
        api_key = st.text_input(
            t(lang, "api_key_label"),
            value=os.getenv("GONKA_API_KEY", DEFAULT_API_KEY),
            type="password",
        )
        # API Key status
        if not api_key.strip():
            st.warning(t(lang, "warn_no_api_key"))
        else:
            is_valid, status = check_api_key(api_key)

            if status == "valid":
                st.success(t(lang, "api_key_connected"))
            elif status == "invalid":
                st.error(t(lang, "api_key_invalid"))
            else:
                st.warning(t(lang, "api_key_check_error"))

        models = st.multiselect(
            t(lang, "models_label"), options=DEFAULT_MODELS, default=DEFAULT_MODELS
        )
        extra = st.text_input(t(lang, "extra_models"), "")
        if extra:
            extras = [m.strip() for m in extra.split(",") if m.strip()]
            models = list(dict.fromkeys(models + extras))
        enable_search = st.toggle(
            t(lang, "enable_search"),
            value=True,
            help=t(lang, "enable_search_help"),
        )

    # ---------------- Main area: input and checking ----------------
    st.title(t(lang, "title"))
    st.caption(t(lang, "caption"))

    user_input = st.text_area(
        t(lang, "input_label"),
        height=150,
        placeholder=t(lang, "input_placeholder"),
    )
    if st.button(t(lang, "check_button"), type="primary", use_container_width=True):
        handle_check(user_input, api_key, models, enable_search, lang)
    elif st.session_state.get("last_consensus") is not None:
        # Keep showing the last result on page refresh/parameter changes
        render_results(st.session_state["last_consensus"],
                       st.session_state["last_results"], lang)


if __name__ == "__main__":
    render_ui()
