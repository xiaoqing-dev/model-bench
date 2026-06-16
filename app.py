"""model-bench — Streamlit UI.

Two modes:
  • 模型对比      — same prompt across many models, judged pairwise.
  • 调 System Prompt — rule-guided system-prompt variants, compared for 人味/温度.

Both support a money-saving funnel (cheap scout round → promote → deep compare).

Run:  streamlit run app.py
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
    generate_system_variants,
    run_funnel,
    run_matrix,
)
from modelbench.aggregate import win_rates
from modelbench.judge import judge_matrix

# Curated whitelist — 3-4 current/flagship models per vendor (verified 2026-06).
CURATED = [
    "openai/gpt-5.5-pro", "openai/gpt-5.5", "openai/gpt-5.4-mini",
    "anthropic/claude-fable-5", "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6", "anthropic/claude-haiku-4.5",
    "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash", "deepseek/deepseek-v3.2",
    "z-ai/glm-5.1", "z-ai/glm-5", "z-ai/glm-4.7",
    "minimax/minimax-m3", "minimax/minimax-m2.7", "minimax/minimax-m2.5",
    "qwen/qwen3.7-max", "qwen/qwen3.7-plus", "qwen/qwen3.6-flash",
    "x-ai/grok-4.3", "x-ai/grok-4.20",
    "google/gemini-3.5-flash", "google/gemini-3.1-pro-preview", "google/gemini-2.5-pro",
    "openrouter/owl-alpha",
]
FLAGSHIP_DEFAULTS = [
    "openai/gpt-5.5-pro", "anthropic/claude-opus-4.8", "deepseek/deepseek-v4-pro",
    "z-ai/glm-5.1", "minimax/minimax-m3", "qwen/qwen3.7-max",
    "x-ai/grok-4.3", "google/gemini-3.1-pro-preview", "openrouter/owl-alpha",
]

_RUBRIC_FILE = Path(__file__).parent / "experiments" / "rubric.md"
WRITING_RUBRIC = _RUBRIC_FILE.read_text() if _RUBRIC_FILE.exists() else "选更好的那一个,相当判 tie。"
HUMAN_RUBRIC = """比较两条「对用户的回复」,选更有人味、更自然流畅、有温度的那条:
- 像真人在说话,不是模板腔 / 客服腔 / AI 腔
- 自然流畅,不生硬、不啰嗦、不翻译腔
- 有温度、有同理心,但不谄媚、不油腻、不过度热情
- 该具体时具体,不空泛正确的废话
更符合以上的为胜;两者相当判 tie。不要因为更长而加分。"""


def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def require_password() -> None:
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
    models = run_async(OpenRouterClient(api_key=api_key).list_models())
    out = []
    for m in models:
        p = m.get("pricing", {}) or {}
        try:
            pin, pout = float(p.get("prompt", 0)) * 1e6, float(p.get("completion", 0)) * 1e6
        except (TypeError, ValueError):
            pin = pout = 0.0
        out.append({"id": m.get("id", ""), "in": pin, "out": pout})
    return sorted(out, key=lambda x: x["id"])


def model_picker(api_key: str):
    """Section 1 — curated model multiselect over the live catalog."""
    st.subheader("选模型")
    top = st.columns([1, 1, 2])
    if top[0].button("刷新模型目录"):
        fetch_catalog.clear()
        st.session_state.pop("catalog", None)
    if "catalog" not in st.session_state:
        st.session_state.catalog = fetch_catalog(api_key)
    catalog = st.session_state.catalog
    all_ids = [m["id"] for m in catalog]
    price = {m["id"]: m for m in catalog}

    show_all = top[1].toggle("显示全部模型", value=False, help="默认只显示精选白名单")
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
    st.session_state.picked = [p for p in st.session_state.picked if p in options]

    picked = st.multiselect(
        "候选模型",
        options=options,
        key="picked",
        format_func=lambda i: f"{i}  (${price[i]['in']:.2f}/${price[i]['out']:.2f} per 1M)",
    )
    if missing and not show_all:
        st.caption("未上架已跳过:" + ", ".join(missing))
    return picked, all_ids, price


def judge_box(all_ids: list, default_rubric: str, picked: list):
    """Judge model + rubric editor."""
    st.subheader("裁判")
    pref = ["anthropic/claude-opus-4.8", "openai/gpt-5.5-pro", "google/gemini-3.1-pro-preview"]
    judge_default = next((m for m in pref if m in all_ids), all_ids[0])
    judge_model = st.selectbox(
        "裁判模型(务必选不在候选集里的强模型)",
        options=all_ids, index=all_ids.index(judge_default),
    )
    rubric = st.text_area("评分标准(rubric)", value=default_rubric, height=180)
    if judge_model in picked:
        st.warning("裁判在候选集里 —— 会自我偏好,建议换一个不参赛的模型。")
    return judge_model, rubric


def parse_json_cases(text: str):
    cases = []
    for n, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(Case(id=f"c{n+1}", vars=json.loads(line)))
        except json.JSONDecodeError:
            st.error(f"第 {n+1} 行不是合法 JSON:{line}")
            st.stop()
    return cases or [Case(id="c1", vars={})]


def show_results(results: list, standings, axis: str) -> None:
    errors = [r for r in results if not r.ok]
    ok = [r for r in results if r.ok]
    total = sum(r.cost_usd or 0 for r in results)
    c1, c2, c3 = st.columns(3)
    c1.metric("单元", len(results))
    c2.metric("失败/空", len(errors))
    c3.metric("成本", f"${total:.4f}")
    if errors:
        with st.expander(f"{len(errors)} 个失败/空输出"):
            for r in errors:
                st.text(f"{r.prompt_id} | {r.model} | {r.case_id}: {r.error}")

    st.markdown("##### 输出对比")
    col_field = "model" if axis == "model" else "prompt_id"
    table: dict = {}
    for r in ok:
        table.setdefault(r.case_id, {})[getattr(r, col_field)] = r.output
    for case_id, row in table.items():
        st.markdown(f"**{case_id}**")
        if not row:
            continue
        cols = st.columns(len(row))
        for col, (label, text) in zip(cols, row.items()):
            with col:
                st.caption(label)
                st.write(text)

    if standings is not None:
        st.markdown("##### 排行榜(swap-tested pairwise 胜率)")
        df = pd.DataFrame(
            [{"参赛方": s.label, "胜率": round(s.win_rate, 3),
              "胜": s.wins, "负": s.losses, "平": s.ties} for s in standings]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


def execute(prompts, models, cases, client, params, judge_model, rubric, *,
            axis, do_judge, use_funnel, top_k, concurrency):
    """Run (funnel or plain) and render."""
    n_candidates = len(models) if axis == "model" else len(prompts)
    if use_funnel and do_judge and n_candidates > top_k:
        with st.spinner("漏斗赛:初赛粗筛 → 决赛深比…"):
            out = run_async(run_funnel(
                prompts, models, cases, client, judge_model, rubric,
                axis=axis, top_k=top_k, params=params,
            ))
        st.success("晋级决赛:" + ", ".join(out["promoted"]))
        st.markdown("### 决赛(全用例深比)")
        show_results(out["round2"]["results"], out["round2"]["standings"], axis)
        with st.expander("初赛(粗筛)明细"):
            show_results(out["round1"]["results"], out["round1"]["standings"], axis)
    else:
        with st.spinner("调用模型中…"):
            results = run_async(run_matrix(prompts, models, cases, client,
                                           concurrency=concurrency, params=params))
        standings = None
        if do_judge:
            with st.spinner("裁判打分中(swap-test)…"):
                outcomes = run_async(judge_matrix(client, judge_model, rubric, results, axis=axis))
            standings = win_rates(outcomes)
        show_results(results, standings, axis)
        st.caption("胜率=胜/(胜+负),平局不计。主观写作的自动分仅作粗筛,务必结合上面人眼对比。")


# ---------------- modes ----------------

def mode_model_compare(api_key, params, do_judge, use_funnel, top_k, concurrency):
    picked, all_ids, _ = model_picker(api_key)

    st.subheader("对比维度")
    axis = st.radio(
        "对比什么", ["model", "prompt"],
        format_func=lambda a: "比模型(一个 prompt,多模型)" if a == "model" else "比 Prompt 版本(一个模型,多 prompt)",
        horizontal=True,
    )
    st.subheader("Prompt 与测试输入")
    n_prompts = 1 if axis == "model" else st.number_input("Prompt 版本数", 2, 6, 2)
    prompts = []
    for i in range(int(n_prompts)):
        t = st.text_area(f"Prompt v{i+1}", height=110, key=f"mc_prompt_{i}",
                         value="用中文写一段 120 字的开头,主题:{{topic}}。从一个具体事件切入,别用大词。")
        prompts.append(PromptVersion(id=f"v{i+1}", template=t))
    cases = parse_json_cases(st.text_area(
        "测试输入(每行一个 JSON;留空=无变量)",
        value='{"topic": "我用 AI 写了第一行生产代码"}\n{"topic": "公开构建一家公司的第一周"}',
        height=90,
    ))

    judge_model, rubric = judge_box(all_ids, WRITING_RUBRIC, picked)

    st.subheader("运行")
    if not picked:
        st.info("先选至少一个候选模型。")
        return
    if st.button("▶ 运行", type="primary"):
        client = OpenRouterClient(api_key=api_key)
        execute(prompts, picked, cases, client, params, judge_model, rubric,
                axis=axis, do_judge=do_judge, use_funnel=use_funnel, top_k=top_k, concurrency=concurrency)


def mode_system_tune(api_key, params, do_judge, use_funnel, top_k, concurrency):
    st.subheader("1 · 基础 system prompt 与规则")
    base = st.text_area("基础 system prompt(要优化的起点)", height=120,
                        value="你是一个助手,回答用户的问题。")
    rules = st.text_area(
        "变体生成规则(你研究的「人味/温度/流畅」原则,逐条写)", height=160,
        value="- 用口语和短句,像朋友聊天,不要书面语和官腔\n"
              "- 先共情/回应情绪,再给信息\n"
              "- 允许适度的不确定和个人语气,不要全知全能腔\n"
              "- 不堆砌列表和小标题,用自然段落",
    )
    gen_col = st.columns([1, 1, 2])
    n_var = gen_col[0].number_input("变体数", 2, 8, 4)

    picked, all_ids, _ = model_picker(api_key)
    gen_default = next((m for m in ["anthropic/claude-opus-4.8", "openai/gpt-5.5-pro"] if m in all_ids), all_ids[0])
    gen_model = st.selectbox("变体生成器模型", options=all_ids, index=all_ids.index(gen_default))

    if gen_col[1].button("✦ 按规则生成变体"):
        with st.spinner("生成 system prompt 变体中…"):
            variants = run_async(generate_system_variants(
                OpenRouterClient(api_key=api_key), gen_model, base, rules, int(n_var)))
        st.session_state.variants = variants

    variants = st.session_state.get("variants", [])
    if not variants:
        st.info("点「按规则生成变体」后,这里会出现可编辑的变体。")
        return

    st.subheader("2 · 生成的变体(可手动改)")
    edited = []
    for i, v in enumerate(variants):
        edited.append(st.text_area(f"变体 v{i+1}", value=v, height=110, key=f"var_{i}"))

    st.subheader("3 · 测试输入(模拟用户消息,每行一条)")
    inputs_raw = st.text_area(
        "每行一条用户消息", height=110,
        value="我最近压力好大,什么都不想做。\n帮我看看这段代码为什么报错。\n推荐几本关于创业的书。",
    )
    inputs = [ln.strip() for ln in inputs_raw.splitlines() if ln.strip()]
    cases = [Case(id=f"in{j+1}", vars={"input": text}) for j, text in enumerate(inputs)] or \
            [Case(id="in1", vars={"input": "你好"})]

    # variants -> prompt versions (user msg = {{input}}, system = variant)
    prompts = [PromptVersion(id=f"v{i+1}", template="{{input}}", system=v) for i, v in enumerate(edited)]

    judge_model, rubric = judge_box(all_ids, HUMAN_RUBRIC, picked)

    st.subheader("4 · 运行(比 system prompt 变体)")
    if not picked:
        st.info("先选至少一个用来测试的模型。")
        return
    if st.button("▶ 运行变体对比", type="primary"):
        client = OpenRouterClient(api_key=api_key)
        # axis="prompt": compare the system-prompt variants (on the picked model(s))
        execute(prompts, picked, cases, client, params, judge_model, rubric,
                axis="prompt", do_judge=do_judge, use_funnel=use_funnel, top_k=top_k, concurrency=concurrency)


def main() -> None:
    st.set_page_config(page_title="model-bench", layout="wide")
    require_password()
    st.title("model-bench")

    with st.sidebar:
        st.header("设置")
        api_key = st.text_input("OpenRouter API Key", value=get_secret("OPENROUTER_API_KEY", ""),
                                type="password", help="部署时从服务器密钥读;本地从 .env 读")
        max_tokens = st.number_input("max_tokens", 256, 32000, 4000, step=256,
                                     help="推理模型的隐藏思考也算在内,太低会导致正文为空。建议 ≥4000。")
        temperature = st.slider("temperature", 0.0, 2.0, 1.0, 0.1)
        concurrency = st.slider("并发数", 1, 16, 8)
        do_judge = st.toggle("跑裁判打分", value=True)
        use_funnel = st.toggle("省钱漏斗(先粗筛再深比)", value=False,
                               help="候选多于 top_k 时:先用部分用例粗筛,只让晋级者进决赛")
        top_k = st.number_input("晋级数 top_k", 2, 8, 3) if use_funnel else 3

    if not api_key:
        st.warning("请在左侧填入 OpenRouter API Key(或设环境变量 / .env)。")
        st.stop()

    params = {"max_tokens": int(max_tokens), "temperature": float(temperature)}
    mode = st.radio("模式", ["模型对比", "调 System Prompt"], horizontal=True)
    st.divider()
    if mode == "模型对比":
        mode_model_compare(api_key, params, do_judge, use_funnel, int(top_k), int(concurrency))
    else:
        mode_system_tune(api_key, params, do_judge, use_funnel, int(top_k), int(concurrency))


if __name__ == "__main__":
    main()
