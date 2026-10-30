# CS527 project: agents on delegated GPU kernel optimization

Task candidates come from `../cs527_task_mining/`. This directory holds the
experiment infrastructure.

```
docker/Dockerfile            task image: the parent commit on its CI image
scripts/build_task.sh        docker build + warm-up on a GPU + docker commit
scripts/eval_correctness.sh  visible + hidden tests and reference checks for one patch
scripts/eval_perf.sh         interleaved, clock-locked timing of several patches
scripts/common.sh            per-architecture image selection, shared mounts
scripts/run_agent(s).py/.sh  one mini-swe-agent run / several runs per arm
scripts/gpu_tools.py         run_tests / bench / profile for GPU-less agent containers
scripts/agent_prompt.yaml    the agent's prompt templates
scripts/eval_runs.sh         evaluate every run of a task; summarize_runs.py makes the table
tasks/<id>/task.env          reference commit, CI image, supported SMs, tests, warm-up
tasks/<id>/task/             agent-visible files, /task in the image
tasks/<id>/eval/             evaluation only: ref.patch, hidden tests and shapes
```

A task runs on any GPU in its `SUPPORTED_SM`; results (baseline, reference and
agent alike) hold for that architecture and are never mixed across
architectures. Images are per architecture, `cs527/<id>:sm90` on H100. So far
c3f7334d4 has been validated on SM90 only; its SM10x path is wired up but
untested.

## Build (on the GPU host, once per architecture)

The scripts call plain `docker` (and mini-swe-agent cannot call it through
`sudo`), so add yourself to the docker group once and log in again:

```bash
sudo usermod -aG docker $USER
scripts/build_task.sh c3f7334d4 0
```

## Run agents

The agent is [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent),
run on the GPU host. The model is queried from the host; the agent's commands
run in an offline container with no GPU, where it edits and compiles. Code
reaches the GPU only through the arm's tools (`scripts/gpu_tools.py`): each
call builds the tree in the agent's container, then runs in a GPU worker
container that mounts the agent's volumes, behind one lock per GPU, so several
agents can share a GPU without overlapping timings.

| arm | tools |
|---|---|
| `correctness` | `run_tests` (pass/fail and errors, timings stripped) |
| `timing` | + `bench` (`/task/bench_task.py`) |
| `ncu-select` | + `profile`, the agent picks ncu sections/metrics |
| `ncu-full` | + `profile`, every ncu section |
| `ncu-summary` | + `profile`, an LLM summary of every ncu section |

One-time setup on the host:

```bash
python3 -m venv ~/mswea && ~/mswea/bin/pip install mini-swe-agent==2.4.6 pyyaml mcp  # litellm needs mcp at run time
export OPENAI_API_KEY=...       # e.g. in ~/.bashrc
```

The host must restrict GPU performance counters to admins
(`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`, set with
`options nvidia NVreg_RestrictProfilingToAdminUsers=1` in `/etc/modprobe.d`), so
code the agent runs through `run_tests` or `bench` cannot profile itself;
`run_agent.py` refuses to start otherwise (`--no-profiling-check` for pilots).

Then, for N attempts per arm (started in alternating arm order; `PARALLEL`
agents at a time share the GPU, each building with cores/`PARALLEL` jobs):

```bash
PARALLEL=3 PYTHON=~/mswea/bin/python scripts/run_agents.sh c3f7334d4 0 3 ~/runs
```

The agent's prompt is `TASK.md`, the arm's tools and the step budget, inside
our templates (`scripts/agent_prompt.yaml`).

`ARMS="timing ncu-full"` limits the arms; `run_agent.py` options go after `--`
(`--model`, default `openai/gpt-6.1-sol`; `--step-limit`, default 150). Each run
directory (`~/runs/<task>/<arm>-<k>/`) holds `final.patch`, per-step snapshots
in `steps/`, `trajectory.json`, `tools.jsonl` (build, queue and GPU seconds per
tool call), `passed/` (the tree at every passing `run_tests`) and `meta.json`
(steps, tokens, cost, wall time, exit status, tool calls, and any mention of
profilers or GPU timing APIs). `scored.patch` is what gets evaluated:
`final.patch` if the agent submitted, otherwise its last passing `run_tests`
snapshot (empty if its tests never passed).

To evaluate all runs of a task (correctness of each `scored.patch`, then one
interleaved perf eval of base, reference and every correct run, then
`summary.md` and `runs.csv`):

```bash
scripts/eval_runs.sh c3f7334d4 ~/runs/c3f7334d4 ~/evals/c3f7334d4 0 5
```

`eval_perf.sh` holds the same per-GPU lock as the agents' tools while timing,
so it can run while agents are still working.

## Evaluate

Evaluation containers are offline, mount `/task` read-only from the host, and
discard patch edits to `tests/`, `benchmarks/` and `conftest.py`.

```bash
scripts/eval_correctness.sh c3f7334d4 none results/base 0
BASE_OUT=results/base scripts/eval_correctness.sh c3f7334d4 tasks/c3f7334d4/eval/ref.patch results/ref 0
scripts/eval_perf.sh c3f7334d4 results/perf 0 5 base=none ref=tasks/c3f7334d4/eval/ref.patch
```

`BASE_OUT` lists every test whose outcome differs from the baseline, so a patch
cannot pass by turning tests into skips. Add `agent=<patch>` as another perf arm
to score an agent run.
