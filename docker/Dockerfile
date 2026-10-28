# check=skip=InvalidDefaultArgInFrom
# Task image: FlashInfer at REF_COMMIT^ on the CI image that commit was tested
# with. Built by scripts/build_task.sh, which passes BASE_IMAGE, REF_COMMIT and
# the `task` context (copied to /task).

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG REF_COMMIT
WORKDIR /workspace/flashinfer

# One RUN, so no layer holds upstream history:
# - fetch REF_COMMIT with depth 2 and check out its parent (refusing a merge commit),
#   with all submodules, then drop every .git;
# - install like CI (scripts/setup_test_env.sh); BUILD_NVEP=0 skips the multi-GPU
#   NIXL-EP build;
# - re-init as a one-commit repo after the install, so generated files are never
#   in the agent's diff.
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

CMD ["sleep", "infinity"]
