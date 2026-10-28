# CS527 project: agents on delegated GPU kernel optimization

Task candidates come from `../cs527_task_mining/`. This directory holds the
experiment infrastructure.

```
docker/Dockerfile            task image: the parent commit on its CI image
scripts/build_task.sh        docker build + warm-up on a GPU + docker commit
scripts/eval_correctness.sh  visible + hidden tests and reference checks for one patch
scripts/eval_perf.sh         interleaved, clock-locked timing of several patches
scripts/common.sh            per-architecture image selection, shared mounts
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

```bash
DOCKER="sudo docker" scripts/build_task.sh c3f7334d4 0
```

## Run an agent container

```bash
sudo docker run -d --name c3f-timing-01 --gpus '"device=0"' --network none --ipc=host cs527/c3f7334d4:sm90
```

Timing only: without GPU performance counters `ncu` fails with
ERR_NVGPUCTRPERM, but only if the host restricts profiling to admins
(`RmProfilingAdminOnly: 1` in `/proc/driver/nvidia/params`; check with
`require_profiling_admin_only` in `common.sh`). Timing + ncu: add
`--cap-add SYS_ADMIN`. Run as root with
`HOME=/root` (the warmed JIT cache). Export the result with
`git add -A && git diff --cached --binary` in `/workspace/flashinfer`.

## Evaluate

Evaluation containers are offline, mount `/task` read-only from the host, and
discard patch edits to `tests/`, `benchmarks/` and `conftest.py`.

```bash
DOCKER="sudo docker" scripts/eval_correctness.sh c3f7334d4 none results/base 0
DOCKER="sudo docker" BASE_OUT=results/base scripts/eval_correctness.sh c3f7334d4 tasks/c3f7334d4/eval/ref.patch results/ref 0
DOCKER="sudo docker" scripts/eval_perf.sh c3f7334d4 results/perf 0 5 base=none ref=tasks/c3f7334d4/eval/ref.patch
```

`BASE_OUT` lists every test whose outcome differs from the baseline, so a patch
cannot pass by turning tests into skips. Add `agent=<patch>` as another perf arm
to score an agent run.
