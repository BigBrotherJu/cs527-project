# Task image: FlashInfer at a task's parent commit, flattened to a single-commit
# repo, editable-installed on top of the CI image that commit was tested with.
#
# Build with scripts/build_task.sh (it passes the args and the task context, then
# warms the JIT cache on a GPU, which docker build cannot do).
#
#   BASE_IMAGE  CI image digest from the parent commit's ci/docker-tags.yml
#   REF_COMMIT  full 40-char hash of the reference (answer) commit; the image holds
#               its parent (GitHub refuses short hashes in fetch)
#   task        build context with the agent-visible task files (TASK.md, bench_task.py, ...)

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG REF_COMMIT
WORKDIR /workspace/flashinfer

# One RUN so no layer ever holds upstream history: fetch the reference commit and
# its parent (fetch takes a hash, not REF_COMMIT^), refuse a merge commit whose ^
# would be ambiguous, check out the parent, fetch all submodules (no hand-kept list that a later parent commit could
# outgrow), drop every .git (the submodules' .git files too), then re-init as a
# one-commit repo the agent's diff is taken against.
#
# Install like CI (scripts/setup_test_env.sh): --no-deps keeps the image's
# preinstalled stack, FLASHINFER_BUILD_NO_PIP=1 stops the build hooks from pip
# installing. BUILD_NVEP=0 skips the best-effort NIXL-EP (multi-GPU MoE EP) build,
# which single-GPU kernel tasks never use.
#
# Install before `git init` so files the install generates (flashinfer/data
# symlinks, _build_meta.py) are in the initial commit or gitignored, never in the
# agent's diff.
RUN git init -q . \
 && git remote add origin https://github.com/flashinfer-ai/flashinfer.git \
 && git fetch -q --depth 2 origin "${REF_COMMIT}" \
 && ! git rev-parse -q --verify 'FETCH_HEAD^2' \
 && git checkout -q 'FETCH_HEAD^' \
 && git submodule update -q --init --recursive --depth 1 \
 && find . -name .git -prune -exec rm -rf {} + \
 && rm -f .gitmodules \
 && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 \
    pip install --no-build-isolation --no-deps -e . \
 && git config --global user.email agent@local \
 && git config --global user.name agent \
 && git config --global --add safe.directory '*' \
 && git init -q \
 && git add -A \
 && git commit -qm "initial"

COPY --from=task . /task/

# Keep the container alive under `docker run -d`; the agent attaches with docker exec.
CMD ["sleep", "infinity"]
