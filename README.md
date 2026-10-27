# CS527 project: agents on delegated GPU kernel optimization

Task candidates come from `../cs527_task_mining/`. This directory holds the
experiment infrastructure.

```
docker/Dockerfile          generic task image (one per task, built with build args)
scripts/build_task.sh      docker build + JIT warm-up on a GPU + docker commit
tasks/<id>/task.env        parent commit, CI image digest, reference commit, warm-up command
tasks/<id>/task/           agent-visible files, copied to /task in the image
```

Evaluation-only material (hidden tests and shapes, the reference patch) is
mounted into evaluation containers and never built into the image.

## Build a task image (on the GPU host)

```bash
DOCKER="sudo docker" scripts/build_task.sh c3f7334d4 0
```

The build fetches only the parent commit and its submodules (shallow),
removes all git metadata, installs FlashInfer like CI does, and
re-initializes `/workspace/flashinfer` as a one-commit repo. The warm-up then
runs the task's tests and benchmark once on GPU 0, so the JIT module is
compiled before the agent starts, and commits the container as
`cs527/<id>:latest`.

## Run a container for an agent

Timing only (no GPU performance counters, so `ncu` fails with ERR_NVGPUCTRPERM):

```bash
sudo docker run -d --name c3f-timing-01 --gpus '"device=0"' --network none --ipc=host cs527/c3f7334d4:latest
```

Timing + ncu: add `--cap-add SYS_ADMIN` (and mount the profiling tool once it exists).

Containers must run as root with the default `HOME=/root`, where the warmed
JIT cache lives. Export the agent's result with
`git add -A && git diff --cached --binary` inside `/workspace/flashinfer`.

## Evaluate a patch

`tasks/<id>/eval/` holds the evaluation-only material: `ref.patch` (the
upstream answer), `hidden_shapes.json` and `test_hidden_*.py`. Evaluation
containers are offline and mount `/task` read-only from the host.
`scripts/in_container/prepare.sh` drops edits to `tests/`, `benchmarks/` and
`conftest.py` (reporting them) and clears the JIT build cache before anything runs.

Correctness (visible + hidden tests, reference check on visible + hidden shapes):

```bash
DOCKER="sudo docker" scripts/eval_correctness.sh c3f7334d4 none results/base 0
DOCKER="sudo docker" scripts/eval_correctness.sh c3f7334d4 tasks/c3f7334d4/eval/ref.patch results/ref 0
```

Performance (one container per arm, arm order rotated each round):

```bash
DOCKER="sudo docker" scripts/eval_perf.sh c3f7334d4 results/perf 0 5 base=none ref=tasks/c3f7334d4/eval/ref.patch
```

Add `agent=<patch>` as a third arm to score an agent run.
