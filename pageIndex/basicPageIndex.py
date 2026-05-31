from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pageindex import PageIndexClient

load_dotenv()

logger = logging.getLogger(__name__)

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DOC_ID = "pi-cmps9bjmz00hr01quulhissiy"
MODEL_NAME = "gpt-4o-mini"
# MODEL_NAME = "gpt-5.4"

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = (
    BASE_DIR
    / "Data"
    / "StudyMaterial"
    / "Complete AI"
    / "AI Agents guidebook.pdf"
)

PI_CLIENT = PageIndexClient(api_key=PAGEINDEX_API_KEY)
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


@lru_cache(maxsize=1)
def _get_pageindex_tree() -> list[dict[str, Any]]:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY is missing in environment.")

    status_result = PI_CLIENT.get_document(DOC_ID)
    if status_result.get("status") != "completed":
        raise RuntimeError(
            f"Document is not ready. Current status: {status_result.get('status')}"
        )

    tree_result = PI_CLIENT.get_tree(DOC_ID, node_summary=True)
    tree = tree_result.get("result")
    if not isinstance(tree, list):
        raise RuntimeError("PageIndex tree could not be loaded.")
    return tree


def _compress_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for node in nodes:
        entry: dict[str, Any] = {
            "node_id": node.get("node_id"),
            "title": node.get("title"),
            "page": node.get("page_index", "?"),
            "summary": (node.get("text") or "")[:150],
        }
        if node.get("nodes"):
            entry["children"] = _compress_tree(node["nodes"])
        out.append(entry)
    return out


def llm_tree_search(query: str, tree: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    compressed_tree = _compress_tree(tree)

    prompt = f"""You are given a query and a document's tree structure (like a Table of Contents).
Your task: identify which node IDs most likely contain the answer to the query.
Think step-by-step and reason through the query.

Query: {query}
Document Tree:
{json.dumps(compressed_tree, indent=2)}

Reply ONLY in this exact JSON format:
{{
    "thinking": "<your step-by-step reasoning>",
    "node_list": ["node_id1", "node_id2"]
}}
"""

    started_at = _utc_now_iso()
    t0 = time.perf_counter()
    response = OPENAI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    ended_at = _utc_now_iso()

    payload = json.loads(response.choices[0].message.content)
    usage = _extract_token_usage(response)
    api_log = {
        "stage": "tree_search",
        "model": MODEL_NAME,
        "query": query,
        "request_started_at": started_at,
        "response_received_at": ended_at,
        "elapsed_ms": elapsed_ms,
        **usage,
    }
    return payload, api_log


def find_nodes_by_ids(tree: list[dict[str, Any]], node_ids: list[str]) -> list[dict[str, Any]]:
    found = []
    node_id_set = set(node_ids)
    for node in tree:
        if node.get("node_id") in node_id_set:
            found.append(node)
        if node.get("nodes"):
            found.extend(find_nodes_by_ids(node["nodes"], node_ids))
    return found


def generate_answer(query: str, nodes: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not nodes:
        return "No relevant information found in the document.", {
            "stage": "answer_generation",
            "model": MODEL_NAME,
            "query": query,
            "request_started_at": _utc_now_iso(),
            "response_received_at": _utc_now_iso(),
            "elapsed_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    context_parts = []
    for node in nodes:
        context_parts.append(
            f"""
SECTION: {node.get('title', 'Unknown')}
PAGE: {node.get('page_index', '?')}

CONTENT:
{node.get('text', 'No content available')}
"""
        )

    context_str = "\n\n" + ("\n" + "-" * 80 + "\n").join(context_parts)

    prompt = f"""
You are an expert document analyst.

Your task is to answer the user's query using ONLY the provided document context.

Rules:
1. Use only information found in the context.
2. Do not make up information.
3. Cite every major claim using:
   (Section: <title>, Page: <page>)
4. If the answer is not present in the context, say so.
5. Write a clear and detailed answer.

User Query:
{query}

Document Context:
{context_str}
"""

    started_at = _utc_now_iso()
    t0 = time.perf_counter()
    response = OPENAI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You answer questions using only the provided document context.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    ended_at = _utc_now_iso()

    usage = _extract_token_usage(response)
    api_log = {
        "stage": "answer_generation",
        "model": MODEL_NAME,
        "query": query,
        "request_started_at": started_at,
        "response_received_at": ended_at,
        "elapsed_ms": elapsed_ms,
        **usage,
    }
    return response.choices[0].message.content.strip(), api_log


def ask_pageindex_question(query: str) -> dict[str, Any]:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Question is required.")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment.")

    # Upload flow from notebook intentionally kept as comment because doc is static.
    # upload_result = PI_CLIENT.submit_document(PDF_PATH)
    # doc_id = upload_result["doc_id"]

    tree = _get_pageindex_tree()
    search_result, search_log = llm_tree_search(cleaned_query, tree)
    node_ids = search_result.get("node_list", [])
    nodes = find_nodes_by_ids(tree, node_ids)
    answer, answer_log = generate_answer(cleaned_query, nodes)

    return {
        "answer": answer,
        "selected_node_ids": node_ids,
        "selected_sections": [
            {"title": node.get("title"), "page_index": node.get("page_index")}
            for node in nodes
        ],
        "api_logs": [search_log, answer_log],
    }
