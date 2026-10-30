"""Run one mini-swe-agent attempt at a task.

    python3 scripts/run_agent.py <task_id> <arm> <out_dir> [--gpu N] [--model M]

The model is queried from the host. The agent's commands run with docker exec
in an offline container that has no GPU; code reaches the GPU only through the
arm's tools (gpu_tools.py), which run it in a GPU worker container sharing the
agent's volumes (workspace, JIT cache, /scratch). Arms:

    correctness   run_tests
    timing        run_tests, bench
    ncu-select    run_tests, bench, profile (the agent picks what ncu collects)
    ncu-full      run_tests, bench, profile (every ncu section)
    ncu-summary   run_tests, bench, profile (an LLM summary of every section)

<out_dir> gets:
  trajectory.json   mini-swe-agent's trajectory
  steps/NNN.patch   the repo diff after step NNN, whenever it changed
  passed/NNN.patch  the repo diff whenever run_tests passed (at step NNN)
  final.patch       the repo diff when the run ended
  scored.patch      what evaluation scores: final.patch if the agent submitted,
                    else the last passed/ snapshot (empty if tests never passed)
  tools.jsonl       one line per tool call: build, queue and GPU seconds
  meta.json         model, arm, steps, tokens, cost, wall time, exit status,
                    scored patch, tool use, and any mention of profilers or
                    timing APIs
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import litellm
import yaml
from minisweagent import __version__, package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.models.litellm_model import LitellmModel

from gpu_tools import ARMS, DOCKER, GpuToolEnvironment, tools_prompt

ROOT = Path(__file__).resolve().parents[1]

# Diff of the working tree against HEAD, untracked files included, built in a
# scratch index so the agent's own git index is left untouched.
DIFF_CMD = ("export GIT_INDEX_FILE=/tmp/snapshot.index; git read-tree HEAD && "
            "git add -A && git diff --cached --binary")

# Ways around the arm's feedback: profilers, and hand-written GPU timing.
WATCH_RE = re.compile(r"\b(ncu|nsys|nvprof|torch\.profiler|cupti|perf_counter|elapsed_time|cuda\.Event|"
                      r"cudaEventElapsedTime|clock64|globaltimer)\b")

SUMMARY_PROMPT = """You are a GPU performance engineer. Below is an Nsight Compute report \
of one launch of a CUDA kernel. Summarize it for the engineer optimizing this kernel: what \
limits its performance, the evidence (metric names and values), and what the numbers suggest \
changing. Be specific and concise (at most 300 words). Do not invent metrics that are not in \
the report."""


def load_task(task_id, gpu, check_profiling):
    """IMAGE plus the task.env variables, via common.sh (which rejects unsupported GPUs).

    GPU workers for run_tests and bench run the agent's code without SYS_ADMIN;
    a host that restricts performance counters to admins keeps that code from
    profiling itself."""
    names = ["IMAGE", "VISIBLE_TESTS", "BUILD_CMD", "PROFILE_KERNEL", "PROFILE_SHAPE"]
    check = "require_profiling_admin_only || exit 1;" if check_profiling else ""
    cmd = f"source scripts/common.sh; load_task {task_id} {gpu} || exit 1; {check} printf '%s\\0' " + \
        " ".join(f'"${n}"' for n in names)
    out = subprocess.run(["bash", "-c", cmd], cwd=ROOT, env={**os.environ, "ROOT": str(ROOT)},
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(out.stderr.strip())
    return dict(zip(names, out.stdout.split("\0")))


def repo_diff(container_id):
    out = subprocess.run([DOCKER, "exec", "-w", "/workspace/flashinfer", container_id, "bash", "-c", DIFF_CMD],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


class SnapshotAgent(DefaultAgent):
    """DefaultAgent that saves the repo diff after every step that changed it."""

    def __init__(self, *args, steps_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_dir = steps_dir
        self.last_diff = ""

    def step(self):
        try:
            return super().step()
        finally:
            diff = repo_diff(self.env.container_id)
            if diff is not None and diff != self.last_diff:
                (self.steps_dir / f"{self.n_calls:03d}.patch").write_text(diff)
                self.last_diff = diff


def token_usage(messages):
    """Model tokens summed over every response the run received, including the
    prompt tokens served from the provider's cache and the reasoning tokens
    (billed as completion tokens), where the provider reports them."""
    total = {"prompt_tokens": 0, "cached_prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    for m in messages:
        usage = m.get("extra", {}).get("response", {})
        usage = (usage.get("usage") if isinstance(usage, dict) else None) or {}
        total["prompt_tokens"] += usage.get("prompt_tokens") or 0
        total["completion_tokens"] += usage.get("completion_tokens") or 0
        total["cached_prompt_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        total["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    return total


def record(a, out, steps_dir, image, agent, env, summary_usage, wall):
    """Write final.patch, scored.patch and meta.json for a finished (or failed) run."""
    final = (repo_diff(env.container_id) if env.container_id else None) or ""
    (out / "final.patch").write_text(final)
    exit_status = agent.messages[-1].get("extra", {}).get("exit_status") if agent.messages else None
    passed = sorted((out / "passed").glob("*.patch"))
    if exit_status == "Submitted":
        scored, scored_from = final, "final.patch"
    elif passed:
        scored, scored_from = passed[-1].read_text(), f"passed/{passed[-1].name}"
    else:
        scored, scored_from = "", None
    (out / "scored.patch").write_text(scored)
    calls = [json.loads(l) for l in (out / "tools.jsonl").read_text().splitlines()] \
        if (out / "tools.jsonl").exists() else []
    # The agent's own text and commands, unescaped: scanning their JSON would
    # match e.g. "\nsys.argv" as nsys.
    texts = []
    for m in agent.messages:
        if m.get("role") == "assistant":
            texts.append(str(m.get("content") or ""))
            for call in m.get("tool_calls") or []:
                try:
                    texts.append(json.loads(call["function"]["arguments"]).get("command", ""))
                except (KeyError, TypeError, ValueError):
                    texts.append(str(call))
    commands = "\n".join(texts)
    watch = {}
    for where, text in (("commands", commands), ("final_patch", final)):
        for term in WATCH_RE.findall(text):
            watch.setdefault(where, {}).setdefault(term, 0)
            watch[where][term] += 1
    meta = {
        "task": a.task_id, "arm": a.arm, "model": a.model, "reasoning_effort": a.reasoning_effort,
        "image": image, "gpu": a.gpu,
        "mini_swe_agent": __version__, "step_limit": a.step_limit,
        "profiling_check": a.profiling_check,
        "steps": agent.n_calls, "tokens": token_usage(agent.messages), "cost": agent.cost,
        "wall_seconds": round(wall), "exit_status": exit_status, "scored_from": scored_from,
        "snapshots": len(list(steps_dir.glob("*.patch"))), "passing_snapshots": len(passed),
        "tool_calls": {t: sum(c["tool"] == t for c in calls) for t in ARMS[a.arm][0]},
        "build_seconds": round(sum(c.get("build_s", 0) for c in calls)),
        "gpu_queue_seconds": round(sum(c.get("wait_s", 0) for c in calls)),
        "gpu_seconds": round(sum(c.get("gpu_s", 0) for c in calls)),
        "summary_usage": summary_usage,
        "watch_terms": watch,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("arm", choices=list(ARMS))
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--model", default="openai/gpt-6.1-sol")
    ap.add_argument("--step-limit", type=int, default=150)
    ap.add_argument("--command-timeout", type=int, default=900, help="seconds per shell command (not tools)")
    ap.add_argument("--reasoning-effort", help="passed to the model (and the ncu-summary summarizer); "
                    "default: the model's own")
    ap.add_argument("--max-jobs", type=int, default=os.cpu_count(),
                    help="parallel compile jobs for this agent's builds (lower it when agents run concurrently)")
    ap.add_argument("--profiling-check", action=argparse.BooleanOptionalAction, default=True,
                    help="refuse to run unless the host restricts performance counters to admins "
                         "(--no-profiling-check skips it, for pilots only)")
    a = ap.parse_args()

    out = a.out_dir.resolve()
    steps_dir = out / "steps"
    steps_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a run
    task_env = load_task(a.task_id, a.gpu, a.profiling_check)
    image = task_env["IMAGE"]
    sm = image.rsplit(":sm", 1)[1]                    # e.g. 90, 100
    # The agent's container has no GPU to detect the target from (e.g. 9.0a).
    build_env = {"FLASHINFER_CUDA_ARCH_LIST": f"{sm[:-1]}.{sm[-1]}a", "FLASHINFER_JIT_DEBUG": "0",
                 "MAX_JOBS": str(a.max_jobs)}

    # Per-run volumes, filled from the image on first mount, shared with the
    # GPU workers; /task comes read-only from the host.
    tag = f"cs527-{re.sub(r'[^a-zA-Z0-9]+', '-', out.name)}-{os.getpid()}"
    volumes = {f"{tag}-ws": "/workspace/flashinfer", f"{tag}-cache": "/root/.cache/flashinfer",
               f"{tag}-scratch": "/scratch", str(ROOT / "tasks" / a.task_id / "task"): "/task:ro"}
    run_args = ["--rm", "--network", "none"]
    for name, path in volumes.items():
        run_args += ["-v", f"{name}:{path}"]

    # Settings come from mini-swe-agent's mini.yaml (the bash tool-call format
    # LitellmModel expects); the agent templates (agent_prompt.yaml), task,
    # limits, container and tools are ours.
    cfg = yaml.safe_load((Path(package_dir) / "config" / "mini.yaml").read_text())
    cfg["agent"].update(yaml.safe_load((ROOT / "scripts" / "agent_prompt.yaml").read_text()))
    cfg["agent"].pop("mode", None)  # interactive mini CLI only
    task = ((ROOT / "tasks" / a.task_id / "task" / "TASK.md").read_text()
            + tools_prompt(a.arm, task_env["PROFILE_SHAPE"])
            + f"\nYou have at most {a.step_limit} responses (steps).\n")
    summary_usage = []

    effort = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}

    def summarize(report):
        resp = litellm.completion(model=a.model, drop_params=True, **effort, messages=[
            {"role": "system", "content": SUMMARY_PROMPT}, {"role": "user", "content": report}])
        usage = {k: getattr(resp.usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        summary_usage.append(usage)
        return resp.choices[0].message.content, usage

    env = GpuToolEnvironment(
        arm=a.arm, gpu=a.gpu, volumes=volumes, task_env=task_env, root=ROOT, build_env=build_env,
        summarize=summarize, log_path=out / "tools.jsonl",
        image=image, cwd="/workspace/flashinfer", run_args=run_args, timeout=a.command_timeout,
        container_timeout="24h",
        env={**cfg.get("environment", {}).get("env", {}), **build_env})
    # litellm may not know this model's price, so cost is recorded as 0 rather than failing.
    cfg["model"]["model_kwargs"] = {**cfg["model"].get("model_kwargs", {}), **effort}
    model = LitellmModel(model_name=a.model, cost_tracking="ignore_errors", **cfg["model"])
    agent = SnapshotAgent(model, env, steps_dir=steps_dir, **{
        **cfg["agent"], "step_limit": a.step_limit, "cost_limit": 0, "output_path": out / "trajectory.json"})

    passed_dir = out / "passed"
    passed_dir.mkdir()

    def save_passing():
        diff = repo_diff(env.container_id)
        if diff is not None:
            (passed_dir / f"{agent.n_calls:03d}.patch").write_text(diff)

    env.on_tests_passed = save_passing

    start = time.time()
    try:
        agent.run(task)
    finally:
        try:
            meta = record(a, out, steps_dir, image, agent, env, summary_usage, time.time() - start)
        finally:
            # Remove the container now (cleanup() only stops it in the
            # background), then its volumes, whatever happened above.
            if env.container_id:
                subprocess.run([DOCKER, "rm", "-f", env.container_id], capture_output=True)
            subprocess.run([DOCKER, "volume", "rm", *[v for v in volumes if not v.startswith("/")]],
                           capture_output=True)
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
