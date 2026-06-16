"""model-bench — Streamlit UI.

Run:  streamlit run app.py
Everything is point-and-click: pick models from the live OpenRouter catalog,
paste prompts, click run, read outputs + leaderboard side by side.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from modelbench import (
    Case,
    OpenRouterClient,
    PromptVersion,
    run_matrix,
)
from modelbench.aggregate import win_rates
from modelbench.judge import judge_matrix

# Curated whitelist — per vendor, 3-4 current/flagship models, picked by
# recency + usage and verified against the live catalog (2026-06). Display order
# below. Anything OpenRouter has since pulled is skipped automatically.
CURATED = [
    # OpenAI
    "openai/gpt-5.5-pro", "openai/gpt-5.5", "openai/gpt-5.4-mini",
    # Anthropic
    "anthropic/claude-fable-5", "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6", "anthropic/claude-haiku-4.5",
    # DeepSeek
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v3.2",
    # Zhipu GLM
    "z-ai/glm-5.1", "z-ai/glm-5", "z-ai/glm-4.7",
    # MiniMax
    "minimax/minimax-m3", "minimax/minimax-m2.7", "minimax/minimax-m2.5",
    # Qwen
    "qwen/qwen3.7-max", "qwen/qwen3.7-plus", "qwen/qwen3.6-flash",
    # xAI
    "x-ai/grok-4.3", "x-ai/grok-4.20",
    # Google
    "google/gemini-3.5-flash", "google/gemini-3.1-pro-preview", "google/gemini-2.5-pro",
    # OpenRouter Owl
    "openrouter/owl-alpha",
]

# One flagship per vendor — pre-selected by default for a clean head-to-head.
FLAGSHIP_DEFAULTS = [
    "openai/gpt-5.5-pro",
    "anthropic/claude-opus-4.8",
    "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.1",
    "minimax/minimax-m3",
    "qwen/qwen3.7-max",
    "x-ai/grok-4.3",
    "google/gemini-3.1-pro-preview",
    "openrouter/owl-alpha",
]

DEFAULT_RUBRIC = (Path(__file__).parent / "experiments" / "rubric.md")
RUBRIC_TEXT = DEFAULT_RUBRIC.read_text() if DEFAULT_RUBRIC.exists() else "选更好的那一个。两者相当判 tie。"


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit Cloud secrets first, then env / .env."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def require_password() -> None:
    """Password gate. If APP_PASSWORD isn't configured (e.g. local), stays open."""
    expected = get_secret("APP_PASSWORD")
    if not expected or st.session_state.get("auth_ok"):
        return
    st.title("model-bench")
    pw = st.text_input("访问密码", type="password")
    if pw and pw == expected:
        st.session_state.auth_ok = True
        st.rerun()
    elif pw:
        st.error("密码错误")
    st.stop()


def run_async(coro):
    return asyncio.run(coro)


@st.cache_data(show_spinner="拉取 OpenRouter 模型目录…")
def fetch_catalog(api_key: str) -> list:
    client = OpenRouterClient(api_key=api_key)
    models = run_async(client.list_models())
    out = []
    for m in models:
        pricing = m.get("pricing", {}) or {}
        try:
            pin = float(pricing.get("prompt", 0)) * 1_000_000
            pout = float(pricing.get("completion", 0)) * 1_000_000
        except (TypeError, ValueError):
            pin = pout = 0.0
        out.append({"id": m.get("id", ""), "in": pin, "out": pout})
    return sorted(out, key=lambda x: x["id"])


def main() -> None:
    st.set_page_config(page_title="model-bench", layout="wide")
    require_password()
    st.title("model-bench")
    st.caption("同一 prompt 比多个模型,或同一模型比多个 prompt 版本 —— 由裁判模型 pairwise 打分。")

    # ---- sidebar: settings ----
    with st.sidebar:
        st.header("设置")
        api_key = st.text_input(
            "OpenRouter API Key",
            value=get_secret("OPENROUTER_API_KEY", ""),
            type="password",
            help="部署时从服务器密钥自动读取;本地从 .env / 环境变量读取",
        )
        max_tokens = st.number_input(
            "max_tokens", 256, 32000, 4000, step=256,
            help="推理模型(GPT-5/Gemini/MiniMax 等)的隐藏思考也算在内,设太低会导致正文为空。建议 ≥4000。",
        )
        temperature = st.slider("temperature", 0.0, 2.0, 1.0, 0.1)
        concurrency = st.slider("并发数", 1, 16, 8)
        do_judge = st.toggle("跑裁判打分", value=True)

    if not api_key:
        st.warning("请在左侧填入 OpenRouter API Key(或设环境变量 OPENROUTER_API_KEY)。")
        st.stop()

    # ---- model picker (curated whitelist over the live catalog) ----
    st.subheader("1 · 选模型")
    top = st.columns([1, 1, 2])
    if top[0].button("刷新模型目录"):
        fetch_catalog.clear()
        st.session_state.pop("catalog", None)
    if "catalog" not in st.session_state:
        st.session_state.catalog = fetch_catalog(api_key)
    catalog = st.session_state.catalog

    all_ids = [m["id"] for m in catalog]
    price = {m["id"]: m for m in catalog}

    show_all = top[1].toggle("显示全部模型", value=False, help="默认只显示精选白名单;打开可从全目录挑")
    options = all_ids if show_all else [s for s in CURATED if s in all_ids]
    missing = [s for s in CURATED if s not in all_ids]

    if "picked" not in st.session_state:
        st.session_state.picked = [m for m in FLAGSHIP_DEFAULTS if m in all_ids]

    b = st.columns(3)
    if b[0].button("每家旗舰(默认)"):
        st.session_state.picked = [m for m in FLAGSHIP_DEFAULTS if m in all_ids]
    if b[1].button("全选当前列表"):
        st.session_state.picked = list(options)
    if b[2].button("清空"):
        st.session_state.picked = []

    # Drop any stale picks not in the current options (e.g. after toggling 显示全部).
    st.session_state.picked = [p for p in st.session_state.picked if p in options]

    picked = st.multiselect(
        "候选模型(默认每家旗舰各一个,可手动增删)",
        options=options,
        key="picked",
        format_func=lambda i: f"{i}  (${price[i]['in']:.2f}/${price[i]['out']:.2f} per 1M)",
    )
    if missing and not show_all:
        st.caption("以下精选项当前未上架,已跳过:" + ", ".join(missing))

    # ---- mode + prompts + cases ----
    st.subheader("2 · 选对比维度")
    axis = st.radio(
        "对比什么",
        ["model", "prompt"],
        format_func=lambda a: "比模型(一个 prompt,多个模型)" if a == "model" else "比 Prompt 版本(一个模型,多个 prompt)",
        horizontal=True,
    )

    st.subheader("3 · Prompt 与测试输入")
    st.caption("模板里可用 {{变量}} 占位;下方每行一个 JSON 当一组测试输入。")

    n_prompts = 1 if axis == "model" else st.number_input("Prompt 版本数", 2, 6, 2)
    prompts = []
    for i in range(int(n_prompts)):
        default = "用中文写一段 120 字的开头,主题:{{topic}}。从一个具体事件切入,别用大词。"
        t = st.text_area(f"Prompt v{i+1}", value=default, height=120, key=f"prompt_{i}")
        prompts.append(PromptVersion(id=f"v{i+1}", template=t))

    cases_raw = st.text_area(
        "测试输入(每行一个 JSON;留空=无变量单次运行)",
        value='{"topic": "我用 AI 写了第一行生产代码"}\n{"topic": "公开构建一家公司的第一周"}',
        height=100,
    )
    cases = []
    for n, line in enumerate(cases_raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(Case(id=f"c{n+1}", vars=json.loads(line)))
        except json.JSONDecodeError:
            st.error(f"第 {n+1} 行不是合法 JSON:{line}")
            st.stop()
    if not cases:
        cases = [Case(id="c1", vars={})]

    # ---- judge ----
    st.subheader("4 · 裁判")
    judge_pref = ["anthropic/claude-opus-4.8", "openai/gpt-5.5-pro", "google/gemini-3.1-pro-preview"]
    judge_default = next((m for m in judge_pref if m in all_ids), all_ids[0])
    judge_model = st.selectbox(
        "裁判模型(务必选不在候选集里的强模型)",
        options=all_ids,
        index=all_ids.index(judge_default),
    )
    rubric = st.text_area("评分标准(rubric)", value=RUBRIC_TEXT, height=180)
    if do_judge and judge_model in picked:
        st.warning("裁判模型在候选集里 —— 会有自我偏好。建议换一个不参赛的模型当裁判。")

    # ---- run ----
    st.subheader("5 · 运行")
    if not picked:
        st.info("先选至少一个候选模型。")
        st.stop()
    if not st.button("▶ 运行", type="primary"):
        st.stop()

    client = OpenRouterClient(api_key=api_key)
    params = {"max_tokens": int(max_tokens), "temperature": float(temperature)}

    with st.spinner("调用模型中…"):
        results = run_async(
            run_matrix(prompts, picked, cases, client, concurrency=int(concurrency), params=params)
        )

    errors = [r for r in results if not r.ok]
    total_cost = sum(r.cost_usd or 0 for r in results)
    c1, c2, c3 = st.columns(3)
    c1.metric("总单元", len(results))
    c2.metric("失败", len(errors))
    c3.metric("总成本", f"${total_cost:.4f}")
    if errors:
        with st.expander(f"{len(errors)} 个失败"):
            for r in errors:
                st.text(f"{r.key}: {r.error}")

    # outputs side by side: rows = case, cols = the varying axis
    st.markdown("#### 输出对比")
    ok = [r for r in results if r.ok]
    col_field = "model" if axis == "model" else "prompt_id"
    table = {}
    for r in ok:
        table.setdefault(r.case_id, {})[getattr(r, col_field)] = r.output
    for case_id, row in table.items():
        st.markdown(f"**{case_id}**")
        cols = st.columns(len(row))
        for col, (label, text) in zip(cols, row.items()):
            with col:
                st.caption(label)
                st.write(text)

    # leaderboard
    if do_judge:
        with st.spinner("裁判打分中(swap-test,双倍调用)…"):
            outcomes = run_async(judge_matrix(client, judge_model, rubric, results, axis=axis))
        standings = win_rates(outcomes)
        st.markdown("#### 排行榜(swap-tested pairwise 胜率)")
        df = pd.DataFrame(
            [
                {"参赛方": s.label, "胜率": round(s.win_rate, 3), "胜": s.wins, "负": s.losses, "平": s.ties}
                for s in standings
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("胜率 = 胜 / (胜+负),平局不计。主观写作的自动分仅作粗筛,务必结合上面的人眼对比。")


if __name__ == "__main__":
    main()
