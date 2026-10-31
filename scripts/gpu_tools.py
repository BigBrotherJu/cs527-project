"""GPU tools for an agent whose container has no GPU.

The agent's container compiles; kernels run only through three commands, which
GpuToolEnvironment intercepts on the host instead of executing them in the
container:

    run_tests   the task's visible tests; pass/fail and errors, no timings
    bench       /task/bench_task.py (median GPU time per shape)
    profile     an Nsight Compute report of the task's kernel on one shape

Each call first builds the agent's tree in its own container (BUILD_CMD, on the
CPU), then waits for the GPU (one host-wide lock per GPU, so concurrent agents
queue and timings never overlap) and runs the command in a fresh GPU worker
container mounting the agent's volumes, so the worker finds the build done.
Which commands exist, and what `profile` returns, depends on the arm.
"""

import json
import re
import shlex
import subprocess
import time

from minisweagent.environments.docker import DockerEnvironment

from task import DOCKER, gpu_lock

BUILD_TIMEOUT = 1800  # seconds per build
GPU_TIMEOUT = 900     # seconds per GPU run (normally ~10): a hung kernel must not hold the GPU

# arm -> (commands it has, profile mode); "summary" profiles like "full" and
# returns an LLM summary instead of the report.
ARMS = {
    "correctness": ({"run_tests"}, None),
    "timing": ({"run_tests", "bench"}, None),
    "ncu-select": ({"run_tests", "bench", "profile"}, "select"),
    "ncu-full": ({"run_tests", "bench", "profile"}, "full"),
    "ncu-summary": ({"run_tests", "bench", "profile"}, "summary"),
}
TOOLS = ("run_tests", "bench", "profile")

# mini-swe-agent's prompt tells agents to prefix commands with a cd; the tools
# always run in the repo, so a leading "cd DIR &&" (or ";") is dropped.
CD_PREFIX_RE = re.compile(r"^\s*(cd\s+\S+\s*(&&|;)\s*)+")
SHELL_CHARS = set("|&;<>`$()")

# pytest wall times ("in 2.29s (0:00:02)"): the correctness arm gets no timing.
PYTEST_TIME_RE = re.compile(r" in \d+(\.\d+)?s( \(\d+:\d\d:\d\d\))?")

TOOLS_DOC = {
    "run_tests": "    run_tests        # the task's tests: pass/fail and error messages",
    "bench": ("    bench            # median GPU time (us) per shape in /task/shapes.json\n"
              "    bench --check    # plus a check against a torch reference\n"
              "    bench --shapes /scratch/my_shapes.json   # your own shapes, same format"),
    "profile": {
        "select": ("    profile [--shape NAME | --shapes FILE] [--section S ...] [--metrics a,b] [--set SET]\n"
                   "                     # Nsight Compute report of one launch of the kernel on one shape\n"
                   "                     # (default {shape}); you choose what to collect\n"
                   "                     # (ncu's --section/--metrics/--set; default: basic)"),
        "full": ("    profile [--shape NAME | --shapes FILE]\n"
                 "                     # Nsight Compute report (all sections) of one launch of the kernel\n"
                 "                     # on one shape (default {shape})"),
        "summary": ("    profile [--shape NAME | --shapes FILE]\n"
                    "                     # summary of an Nsight Compute report (all sections) of one launch\n"
                    "                     # of the kernel on one shape (default {shape}), written by a\n"
                    "                     # performance engineer"),
    },
}


def tools_prompt(arm, profile_shape):
    """The Tools section of the task prompt for this arm."""
    tools, mode = ARMS[arm]
    lines = [TOOLS_DOC[t][mode].format(shape=profile_shape) if t == "profile" else TOOLS_DOC[t]
             for t in TOOLS if t in tools]
    saved = "\nThe full report is also saved under /scratch.\n" if mode in ("select", "full") else ""
    return f"""
## Tools

Your shell has no GPU. You can edit and compile here, but code runs on a GPU
only through the commands below. Each builds your current tree, runs on a GPU
and prints the result. Run each as a command of its own (a leading `cd DIR &&`
is fine): not in a pipeline, after other commands, or from a script. Put any
files they read under /scratch.

{chr(10).join(lines)}
{saved}"""


class GpuToolEnvironment(DockerEnvironment):
    def __init__(self, *, arm, gpu, volumes, task, root, build_env, summarize, log_path, **kwargs):
        """build_env: variables for builds and GPU workers (target arch, MAX_JOBS, ...)."""
        super().__init__(**kwargs)
        self.arm, self.gpu, self.volumes, self.task = arm, gpu, volumes, task
        self.root, self.build_env, self.summarize, self.log_path = root, build_env, summarize, log_path
        self.on_tests_passed = lambda: None  # set by the runner: snapshot the passing tree
        # Placeholders for the arm's tools, reached only when a tool is called
        # in the middle of a shell command, which the host cannot intercept.
        for tool in ARMS[arm][0]:
            script = (f"#!/bin/sh\necho '{tool}: run it as a command of its own (a leading cd DIR && is fine), "
                      f"not after other commands, in a pipeline or in a script' >&2\nexit 2\n")
            subprocess.run([DOCKER, "exec", "-i", self.container_id, "sh", "-c",
                            f"cat > /usr/local/bin/{tool} && chmod +x /usr/local/bin/{tool}"],
                           input=script, text=True, capture_output=True)

    def execute(self, action, cwd="", *, timeout=None):
        command = action.get("command", "") if isinstance(action, dict) else str(action)
        rest = CD_PREFIX_RE.sub("", command).strip()
        try:
            argv = shlex.split(rest)
        except ValueError:
            argv = []
        if argv and argv[0] in TOOLS:
            if SHELL_CHARS & set(rest):
                return _result(f"{argv[0]}: run it as a command of its own, without pipes, &&, ; "
                               "or redirection (a leading `cd DIR &&` is fine)", 2)
            return self._tool(argv)
        return super().execute(action, cwd, timeout=timeout)

    def _tool(self, argv):
        if argv[0] not in ARMS[self.arm][0]:
            return _result(f"bash: {argv[0]}: command not found", 127)
        rec = {"tool": argv[0], "args": argv[1:]}
        # A failing tool (timeout, API error) is reported to the agent, never
        # allowed to end the run.
        try:
            return self._run_tool(argv, rec)
        except subprocess.TimeoutExpired as e:
            self._log(rec, 124)
            return _result(f"{argv[0]}: timed out after {e.timeout} s", 124)
        except Exception as e:  # noqa: BLE001
            self._log({**rec, "error": repr(e)}, 1)
            return _result(f"{argv[0]}: tool error: {e!r}", 1)

    def _run_tool(self, argv, rec):
        name = argv[0]
        mode = ARMS[self.arm][1]
        t0 = time.time()

        build = subprocess.run(
            [DOCKER, "exec", "-w", "/workspace/flashinfer", *self._env_args(), self.container_id,
             "bash", "-lc", self.task.config["build_cmd"]],
            capture_output=True, text=True, timeout=BUILD_TIMEOUT)
        rec["build_s"] = round(time.time() - t0, 1)
        if build.returncode:
            self._log(rec, build.returncode)
            return _result("build failed:\n" + (build.stdout + build.stderr)[-8000:], build.returncode)

        if name == "run_tests":
            inner = f"pytest -q -p no:cacheprovider -rfE --tb=short {self.task.visible_tests}"
        elif name == "bench":
            inner = shlex.join(["python", "/task/bench_task.py", *argv[1:]])
        else:
            profile = self.task.config["profile"]
            profile_mode = "full" if mode == "summary" else mode
            extra = ["--save"] if mode != "summary" else []
            if not {"--shape", "--shapes"} & {x.split("=")[0] for x in argv[1:]}:
                extra += ["--shape", profile["shape"]]
            inner = shlex.join(["python", "/eval/bin/profile_task.py", profile_mode,
                                profile["kernel"], *argv[1:], *extra])

        with gpu_lock(self.gpu) as waited:
            rec["wait_s"] = round(waited, 1)
            t2 = time.time()
            worker = f"{self.container_id[:12]}-gpu-{int(t2)}"
            try:
                run = subprocess.run([DOCKER, "run", "--rm", "--name", worker, *self._worker_args(name == "profile"),
                                      "bash", "-lc", inner], capture_output=True, text=True, timeout=GPU_TIMEOUT)
            finally:
                # Killing the docker client leaves the container (and its kernel)
                # running; remove it before releasing the GPU.
                subprocess.run([DOCKER, "rm", "-f", worker], capture_output=True)
            rec["gpu_s"] = round(time.time() - t2, 1)
        out = run.stdout + run.stderr

        if name == "run_tests":
            out = PYTEST_TIME_RE.sub("", out)
            if run.returncode == 0:
                self.on_tests_passed()
        elif name == "profile" and mode == "summary":
            out, rec["summary_usage"] = self.summarize(out)
        self._log(rec, run.returncode)
        return _result(out, run.returncode)

    def _worker_args(self, profile):
        args = ["--gpus", f"device={self.gpu}", "--network", "none", "--ipc=host", "-w", "/workspace/flashinfer",
                *self._env_args(), "-v", f"{self.root}/scripts/in_container:/eval/bin:ro"]
        for name, path in self.volumes.items():
            args += ["-v", f"{name}:{path}"]
        if profile:
            args += ["--cap-add", "SYS_ADMIN"]  # GPU performance counters
        return [*args, self.config.image]

    def _env_args(self):
        return [a for k, v in self.build_env.items() for a in ("-e", f"{k}={v}")]

    def _log(self, rec, returncode):
        rec["returncode"] = returncode
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def _result(output, returncode):
    return {"output": output, "returncode": returncode, "exception_info": ""}
