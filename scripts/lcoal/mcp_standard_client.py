"""Validate the standard MCP Streamable HTTP controller endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


DEFAULT_URL = "http://127.0.0.1:17892/mcp"
REQUIRED_TOOLS = {
    "create_task",
    "start_task",
    "get_task",
    "wait_task",
    "run_task",
    "sign_assets",
    "sign_asset_urls",
    "asr_transcribe",
    "object_detect",
    "tts_synthesize",
    "list_models",
    "get_model",
    "add_model",
    "upsert_model",
    "download_model",
    "enable_model",
    "disable_model",
    "list_nodes",
    "get_cluster_status",
    "list_assets",
    "search_assets",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate lcoal standard MCP Streamable HTTP endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"MCP URL (default: {DEFAULT_URL})")
    args = parser.parse_args(argv)
    try:
        summary = asyncio.run(run(args.url))
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".")[0] == "mcp":
            print(
                json.dumps(
                    {
                        "ok": False,
                        "url": args.url,
                        "error": "Python package `mcp` is not installed in this interpreter",
                        "hint": "Install the official MCP Python SDK, e.g. `pip install mcp`, or run with an environment that already has it.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 1
        raise
    except Exception as exc:
        print(json.dumps({"ok": False, "url": args.url, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if summary.get("ok") else 1


async def run(url: str) -> dict[str, Any]:
    from mcp import ClientSession

    try:
        from mcp.client.streamable_http import streamable_http_client
    except ImportError:
        from mcp.client.streamable_http import streamablehttp_client as streamable_http_client

    async with streamable_http_client(url) as (read_stream, write_stream, *_):
        async with ClientSession(read_stream, write_stream) as session:
            initialize_result = await session.initialize()
            tools_result = await session.list_tools()
            tool_names = sorted(tool.name for tool in tools_result.tools)
            missing = sorted(REQUIRED_TOOLS - set(tool_names))
            if missing:
                return {
                    "ok": False,
                    "url": url,
                    "initialized": to_jsonable(initialize_result),
                    "tool_count": len(tool_names),
                    "tools": tool_names,
                    "missing_tools": missing,
                }

            list_models_result = await session.call_tool("list_models", {})
            cluster_status_result = await session.call_tool("get_cluster_status", {})
            list_nodes_result = await session.call_tool("list_nodes", {})
            list_assets_result = await session.call_tool("list_assets", {})

            return {
                "ok": True,
                "url": url,
                "initialized": to_jsonable(initialize_result),
                "tool_count": len(tool_names),
                "tools_sample": tool_names[:10],
                "required_tools_present": sorted(REQUIRED_TOOLS),
                "list_models": parse_tool_payload(list_models_result),
                "get_cluster_status": parse_tool_payload(cluster_status_result),
                "list_nodes": parse_tool_payload(list_nodes_result),
                "list_assets": parse_tool_payload(list_assets_result),
            }


def parse_tool_payload(result: Any) -> Any:
    if getattr(result, "isError", None) or getattr(result, "is_error", None):
        return {"is_error": True, "content": to_jsonable(getattr(result, "content", []))}
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return to_jsonable(structured)
    content = getattr(result, "content", [])
    if not content:
        return None
    first = content[0]
    text = getattr(first, "text", None)
    if text is None:
        return to_jsonable(first)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
