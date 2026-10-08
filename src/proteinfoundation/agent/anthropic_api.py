#!/usr/bin/env python3
"""Anthropic API backend for the research and expand passes.

Replaces the `claude -p` (Max subscription) path so these calls bill against an
API key instead of session limits. Two subcommands mirroring the two passes:

    research  — server-side web search, produces a target dossier
    expand    — no tools, produces design blocks from a dossier + prior results

Both take the system prompt and user message as files (they are long and
contain newlines, so argv is not a safe channel) and write raw text to
--output.

Usage:
    python anthropic_api.py research \
        --model-tag o50 \
        --system-file prompts/research_system.txt \
        --user-text "PD-L1 (programmed death-ligand 1, CD274, PDB 5o45)" \
        --output research_pool.md

    python anthropic_api.py expand \
        --model-tag o50 \
        --system-file /tmp/system.txt \
        --user-file /tmp/user.txt \
        --output expand_raw.md

The API key is read from ANTHROPIC_API_KEY, falling back to the file named by
ANTHROPIC_KEY_FILE (default `~/.anthropic_key`) -- see `resolve_api_key`. It is
never accepted on the command line, where it would leak into process listings
and shell history.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic


def resolve_api_key() -> str:
    """Return the API key from the environment, else from the key FILE.

    The key is never accepted as an argument -- that would leak it into process
    listings and shell history.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    path = Path(os.environ.get("ANTHROPIC_KEY_FILE",
                               str(Path.home() / ".anthropic_key")))
    if path.is_file():
        # The file is shell: `export ANTHROPIC_API_KEY=sk-...` (quoted or not).
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "ANTHROPIC_API_KEY" not in line:
                continue
            _, _, val = line.partition("=")
            key = val.strip().strip('"').strip("'")
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key
                return key
    raise SystemExit(
        "no Anthropic API key: ANTHROPIC_API_KEY is unset and "
        f"{path} does not define it. Export the key or point "
        "ANTHROPIC_KEY_FILE at a file that does.")

# Model tag -> model ID.
MAX_TOOL_TURNS = 40

#: Beta gate for the `fallbacks: "default"` scalar form. The ARRAY form
#: (`fallbacks: [{...}]`) uses `server-side-fallback-2026-06-01` instead, and
#: pairing either header with the other form is a 400 -- do not "correct" one to
#: match the other.
SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: Model tags that accept `fallbacks`. NOT a superset of MODEL_MAP: measured
#: 2026-09-03, `claude-opus-4-8` returns
#: `400 'claude-opus-4-8' does not support the 'fallbacks' parameter`, and o46/
#: o47 predate it. Sending it unconditionally would 400 every benchmark-loop
#: call, which all run on o46.
FALLBACK_MODELS = frozenset({"o50", "f50"})

MODEL_MAP = {
    "o46": "claude-opus-4-6",
    "o47": "claude-opus-4-7",
    "o48": "claude-opus-4-8",
    "o50": "claude-opus-5",
    # Fable 5. Thinking is ALWAYS ON: `{"type": "adaptive"}` is accepted and an
    # explicit `{"type": "disabled"}` is a 400, so the adaptive default below is
    # the only shape that works. Requires 30-day data retention -- an org below
    # that gets a 400 on every request regardless of payload, which is what a
    # smoke test is for.
    "f50": "claude-fable-5",
}

# Server-side web search. Dynamic filtering is built into this tool version on
# Opus 4.6+ — do not also declare code_execution, which would give the model a
# second execution environment.
WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}

# The server-side tool loop stops with stop_reason="pause_turn" when it hits its
# iteration limit; resending resumes it. Bound the resumes so a pathological
# loop cannot run unbounded.
MAX_RESUMES = 10

# Cost controls. Input tokens dominate: every resume resends the whole
# conversation, and accumulated web-search results make that history large, so
# spend grows super-linearly in search count. Three levers, in order of impact:
#
#   1. Prompt caching  — each resume reads the prior history at ~0.1x instead of
#      full price. Top-level cache_control auto-places the breakpoint on the last
#      cacheable block, which avoids rebuilding (and risking corruption of)
#      thinking and server-tool blocks that must be replayed unmodified.
#   2. --max-searches   — caps searches, bounding both the per-search charge and
#      the history growth that drives input cost.
#   3. --effort         — lower effort means fewer, more-consolidated tool calls.
#
# Caching does nearly all the work: on PDL1 it cut cost from ~$5.92 to ~$0.33.
# Capping searches saves little on top of that but measurably thins the dossier
# (PDL1 at max_uses=20/medium: 7,243 chars, 9 PDB IDs, 3 BSA values, versus
# 11,204 / 23 / 16 uncapped at high effort). So searches are uncapped and effort
# is high by default; the flags remain for cost-constrained runs. Keep whatever
# values you choose identical across model tags, or the model contrast is
# confounded with search budget.
DEFAULT_MAX_SEARCHES = None
DEFAULT_EFFORT = "high"


def _extract_text(content_blocks) -> str:
    """Concatenate the text blocks of a response, ignoring tool/thinking blocks."""
    return "".join(b.text for b in content_blocks if b.type == "text")


def _extract_trace(content_blocks) -> list:
    """Reasoning and search activity, for audit only.

    The dossier itself is the text blocks. This captures what the model thought
    and what it searched, which is what you need to tell a correctly-identified
    target from a confidently-wrong one — the `bp_NTM1A` netrin-1 error would
    have been visible here. Written to a sidecar file that the expand pass never
    reads, so it cannot leak into the generated prompts.
    """
    records = []
    for block in content_blocks:
        kind = getattr(block, "type", "")
        if kind == "thinking":
            records.append({"type": "thinking", "text": getattr(block, "thinking", "")})
        elif kind == "redacted_thinking":
            records.append({"type": "redacted_thinking"})
        elif kind == "server_tool_use":
            records.append({
                "type": "search_query",
                "name": getattr(block, "name", None),
                "input": getattr(block, "input", None),
            })
        elif kind == "web_search_tool_result":
            content = getattr(block, "content", None)
            results = []
            if isinstance(content, list):
                for item in content:
                    results.append({
                        "title": getattr(item, "title", None),
                        "url": getattr(item, "url", None),
                    })
            else:
                results = [{"error": str(getattr(content, "error_code", content))}]
            records.append({"type": "search_result", "results": results})
    return records


#: Attempts and base backoff for `_with_overload_retry`. 6 attempts at 20s
#: doubling covers ~10 min of capacity pressure; the epitope stage that provoked
#: this recovered on a hand retry 90 seconds later.
OVERLOAD_ATTEMPTS = 6
OVERLOAD_BASE_SLEEP = 20.0


def _with_overload_retry(call, attempts=OVERLOAD_ATTEMPTS,
                         base_sleep=OVERLOAD_BASE_SLEEP, sleep=time.sleep):
    """Retry `call` through 529 `overloaded_error`, with exponential backoff.

    The SDK's own `max_retries` did NOT cover this. Measured 2026-09-03: two
    consecutive campaign launches died on the FIRST epitope request with 529,
    each returning in well under a second -- no backoff was spent, so the SDK
    declined to retry (a 529 can carry `x-should-retry: false`, which overrides
    the >=500 rule). A hand retry of the identical call 90 seconds later
    succeeded in 39 s, and a minimal o50 probe in between returned in 3.4 s, so
    the API was not down; that shape of request was briefly refused capacity.

    One 529 must not cost a round. Anything that is not an overload is re-raised
    immediately -- a 400 or a refusal is a real failure and retrying it would
    only bury it.
    """
    for attempt in range(attempts):
        try:
            return call()
        except anthropic.APIStatusError as exc:
            if getattr(exc, "status_code", None) != 529 or attempt == attempts - 1:
                raise
            wait = base_sleep * (2 ** attempt)
            print(f"  529 overloaded; retry {attempt + 1}/{attempts - 1} "
                  f"in {wait:.0f}s", file=sys.stderr, flush=True)
            sleep(wait)
    raise AssertionError("unreachable")


def _run(model_tag, system, user, max_tokens, use_web_search, max_searches,
         effort, local_tools=None):
    """Run one request to completion, resuming across pause_turn boundaries.

    `local_tools` is an optional `target_tools.TargetTools`. When present the
    request carries client-side structure-query tools and this loop also services
    `stop_reason == "tool_use"`: the assistant turn is appended verbatim, each
    tool is executed here, and the results are sent back as a user turn. That is
    what lets the expansion agent inspect the target before choosing an epitope.
    Every call is recorded in the trace sidecar -- under a tool loop the calls
    ARE the record of what the agent knew when it designed.
    """
    model = MODEL_MAP[model_tag]
    # Default SDK timeout is 10 min per request. A research turn with 40+ web
    # searches routinely runs longer, and bp_IDI2 and H1 each failed twice with
    # APITimeoutError while shorter targets succeeded — the timeout, not the
    # target. 30 min per request, and the pause_turn loop above already bounds
    # total work at MAX_RESUMES.
    # `max_retries` was 3. Measured 2026-09-03: a 529 `overloaded_error` burst
    # outlasted it and killed the epitope stage on BOTH targets within a minute
    # of each other -- an outage on Anthropic's side, nothing wrong locally.
    # These calls are long single-shot tool-loop conversations: a retry that
    # arrives too late costs the whole conversation, and the SDK's exponential
    # backoff means the extra attempts cost nothing when the API is healthy.
    client = anthropic.Anthropic(api_key=resolve_api_key(),
                                 timeout=1800.0, max_retries=8)

    # `thinking.display` DEFAULTS TO "omitted" ON o47/o48/o50 and to "summarized"
    # on o46. That silent default change is why every campaign trace written on
    # o50 has an empty `thinking` field while the o46 benchmark loops have full
    # text through the identical `_extract_trace` code: measured 2026-09-03,
    # 514/514 campaign thinking blocks empty against 168/168 populated on the
    # loops. The blocks were always there; only their content was suppressed.
    #
    # Ask for the summary explicitly. Note this recovers a SUMMARY, not the raw
    # chain of thought -- the raw reasoning is never returned on these models, so
    # a trace is evidence of what the agent concluded, not a verbatim transcript.
    #
    # Set only where the field exists: `display` arrived with Opus 4.7, and o46
    # already defaults to "summarized", so sending it there would risk a 400 to
    # request the behavior that model has anyway.
    thinking: dict = {"type": "adaptive"}
    if model_tag != "o46":
        thinking["display"] = "summarized"

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        # Adaptive thinking: on by default for Opus 5, explicit for 4.6/4.7/4.8.
        # budget_tokens is removed on all of these and would 400.
        "thinking": thinking,
        # Auto-placed cache breakpoint. On resume the prefix (system + tools +
        # every prior turn) is served from cache at ~0.1x, which is where nearly
        # all of the saving comes from.
        "cache_control": {"type": "ephemeral"},
        "output_config": {"effort": effort},
        # SERVER-SIDE REFUSAL FALLBACK. Safety classifiers can decline a request
        # in this domain: measured 2026-09-03 on a de novo binder-design prompt,
        # `claude-fable-5` and `claude-opus-4-8` BOTH returned
        # `stop_reason: refusal, category=bio` (o47 and o50 answered). o50 has
        # never refused across 71 campaign_exp_v0 calls, but a refusal raises out
        # of this function, and in a 5-thread expand array one refusal silently
        # costs that thread its 50 designs on an unattended 13-hour run.
        #
        # `"default"` routes by refusal category rather than pinning a substitute
        # model, so there is no fallback model to migrate when one is deprecated.
        # Claude API only -- not available on Bedrock/Vertex/Foundry, which this
        # pipeline does not use. Gated: see FALLBACK_MODELS.
    }
    use_fallbacks = model_tag in FALLBACK_MODELS
    if use_fallbacks:
        kwargs["fallbacks"] = "default"
    if use_web_search:
        tool = dict(WEB_SEARCH_TOOL)
        if max_searches is not None:
            tool["max_uses"] = max_searches
        kwargs["tools"] = [tool]
    if local_tools is not None:
        import target_tools
        # A tools object may carry its own `SCHEMAS`/`dispatch` to expose more
        # than one tool family in a single pass. `TargetTools` defines neither,
        # so it takes the original path unchanged.
        kwargs["tools"] = kwargs.get("tools", []) + list(
            getattr(local_tools, "SCHEMAS", None) or target_tools.SCHEMAS)

    messages = [{"role": "user", "content": user}]
    text_parts = []
    trace = []
    usage = {"input": 0, "output": 0, "searches": 0, "cache_write": 0, "cache_read": 0}

    # The _20260209 search tool runs code execution under the hood for dynamic
    # filtering. If a turn pauses while code-execution work is still pending,
    # the resume must name the same container or the API rejects it with
    # "container_id is required when there are pending tool uses generated by
    # code execution with tools". Carry the container across resumes.
    container_id = None

    # Tool turns and pause_turn resumes share one budget. MAX_TOOL_TURNS is
    # generous because the whole point is to let the agent look properly; the
    # bound exists so a loop that never converges cannot bill forever.
    limit = MAX_RESUMES + (MAX_TOOL_TURNS if local_tools is not None else 0)
    for _ in range(limit + 1):
        if container_id is not None:
            kwargs["container"] = container_id
        # `fallbacks` is a beta parameter, so it needs the beta endpoint. Models
        # that do not accept it keep the ORIGINAL non-beta call path unchanged --
        # o46 runs every benchmark loop and must not be perturbed by a campaign
        # feature.
        def _create():
            if use_fallbacks:
                return client.beta.messages.create(
                    messages=messages, betas=[SERVER_SIDE_FALLBACK_BETA], **kwargs)
            return client.messages.create(messages=messages, **kwargs)

        response = _with_overload_retry(_create)

        container = getattr(response, "container", None)
        if container is not None and getattr(container, "id", None):
            container_id = container.id

        usage["input"] += response.usage.input_tokens
        usage["output"] += response.usage.output_tokens
        usage["cache_write"] += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        usage["cache_read"] += getattr(response.usage, "cache_read_input_tokens", 0) or 0
        server_use = getattr(response.usage, "server_tool_use", None)
        if server_use is not None:
            usage["searches"] += getattr(server_use, "web_search_requests", 0) or 0

        # A refusal returns HTTP 200 with an empty or partial content array —
        # check before reading content so this surfaces as a clear error rather
        # than a silently short dossier.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            category = getattr(detail, "category", None) if detail else None
            raise RuntimeError(f"request refused (category={category})")

        text_parts.append(_extract_text(response.content))
        trace.extend(_extract_trace(response.content))

        if response.stop_reason == "tool_use" and local_tools is not None:
            import target_tools
            results = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                own = getattr(local_tools, "dispatch", None)
                payload = (own(block.name, dict(block.input or {})) if own
                           else target_tools.dispatch(
                               local_tools, block.name, dict(block.input or {})))
                # The RESULT, not just its length. The trace exists to answer
                # "what did the agent know when it chose this epitope"; a call
                # log without its answer cannot. Truncated because
                # surface_residues can return 120 rows, and the full payload is
                # reproducible from the tool anyway.
                trace.append({"type": "local_tool", "name": block.name,
                              "input": dict(block.input or {}),
                              "result_chars": len(payload),
                              "result": payload[:4000],
                              "result_truncated": len(payload) > 4000})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": payload})
            if not results:
                raise RuntimeError("stop_reason=tool_use with no tool_use block")
            messages = messages + [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": results},
            ]
            # A tool turn emits no design text; keep only real output so a
            # partial preamble cannot be mistaken for the answer.
            text_parts.pop()
            continue

        if response.stop_reason != "pause_turn":
            if response.stop_reason == "max_tokens":
                print(
                    f"WARNING: hit max_tokens ({max_tokens}) — output may be truncated",
                    file=sys.stderr,
                )
            break

        # Paused mid-turn: append the assistant turn verbatim (thinking and
        # server-tool blocks must be passed back unmodified) and resend.
        messages = messages + [{"role": "assistant", "content": response.content}]
    else:
        raise RuntimeError(f"did not finish within {limit} turns "
                           f"(resumes + tool calls)")

    return "".join(text_parts), usage, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    for name in ("research", "expand"):
        p = sub.add_parser(name)
        p.add_argument("--model-tag", required=True, choices=sorted(MODEL_MAP))
        p.add_argument("--system-file", required=True, help="File holding the system prompt")
        p.add_argument("--user-file", help="File holding the user message")
        p.add_argument("--user-text", help="User message inline (short prompts only)")
        p.add_argument("--output", required=True, help="Where to write the raw text response")
        p.add_argument("--max-tokens", type=int, default=16000)
        p.add_argument("--effort", default=DEFAULT_EFFORT,
                       choices=["low", "medium", "high", "max"],
                       help=f"Reasoning effort (default: {DEFAULT_EFFORT})")
        if name == "research":
            p.add_argument("--max-searches", type=int, default=DEFAULT_MAX_SEARCHES,
                           help="Cap on web_search calls (default: uncapped; 0 also uncapped)")
        if name == "expand":
            # Structure-query tools. All three must be given together: the
            # numbering the agent sees has to be the numbering hotspot_residues
            # is later validated against, and that comes from target_input.
            p.add_argument("--tools-target-pdb",
                           help="Target PDB; enables the structure-query tools")
            p.add_argument("--tools-input-spec",
                           help="target_input, e.g. A1-169")
            p.add_argument("--tools-pdb-id",
                           help="pdb_id, for known_interfaces()")

    args = parser.parse_args()

    if bool(args.user_file) == bool(args.user_text):
        parser.error("provide exactly one of --user-file or --user-text")

    system = Path(args.system_file).read_text()
    user = Path(args.user_file).read_text() if args.user_file else args.user_text

    max_searches = getattr(args, "max_searches", None)
    if max_searches == 0:
        max_searches = None

    local_tools = None
    if getattr(args, "tools_target_pdb", None):
        if not args.tools_input_spec:
            parser.error("--tools-target-pdb requires --tools-input-spec")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from target_tools import TargetTools
        local_tools = TargetTools(args.tools_target_pdb, args.tools_input_spec,
                                  getattr(args, "tools_pdb_id", None))

    text, usage, trace = _run(
        model_tag=args.model_tag,
        system=system,
        user=user,
        max_tokens=args.max_tokens,
        use_web_search=(args.stage == "research"),
        max_searches=max_searches,
        effort=args.effort,
        local_tools=local_tools,
    )

    if not text.strip():
        raise RuntimeError("empty response text")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    # Audit sidecar: thinking and search activity, next to the dossier but under
    # a different name. Downstream reads `research_pool.md` by exact filename, so
    # this never reaches the expand prompt — it exists to answer "how did the
    # model decide what this target is" after the fact.
    # Never let the audit sidecar fail the run. The dossier is already on disk and
    # cost 6-20 minutes of billed API time; research_dossier.sh deletes its
    # .partial on any non-zero exit, so an unserialisable trace field or a full
    # filesystem here would throw away the deliverable.
    if trace:
        try:
            trace_path = out.with_name(out.stem + ".trace.jsonl")
            with open(trace_path, "w") as handle:
                for record in trace:
                    handle.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:  # noqa: BLE001 - audit output is best-effort
            print(f"WARNING: could not write trace sidecar: {exc}", file=sys.stderr)

    # Billable input is uncached + cache writes (~1.25x) + cache reads (~0.1x).
    # Printing all three makes it obvious when caching is silently not working:
    # a cache_read of 0 across resumes means the prefix is being re-billed.
    print(
        f"  {args.stage} [{args.model_tag}]: {len(text)} chars, "
        f"{usage['input']} in / {usage['output']} out, "
        f"cache {usage['cache_write']} write / {usage['cache_read']} read, "
        f"{usage['searches']} searches",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
