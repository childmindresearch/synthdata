#!/usr/bin/env bash
# Scheduled test run for a Linux GPU box (e.g. weekly from cron): the tiers CI
# skips. It runs the sidecar generators (TabularARGN, CDTD) in their own
# environments, failing instead of skipping when one is missing, and the golden
# regression check of the fixture's metrics.
#
#   scripts/scheduled_tests.sh            # from the repo root, on the commit to test
#
# Cron example (Mondays 03:00, log kept next to the checkout):
#   0 3 * * 1 cd ~/synthdata && git pull --ff-only && scripts/scheduled_tests.sh >> ~/synthdata-scheduled.log 2>&1
#
# Exit code: 0 when both tiers pass. A golden failure after an intended change
# means the reference needs re-recording (tests/integration/README.md).
set -uo pipefail

cd "$(dirname "$0")/.."
echo "== $(date -Is) scheduled tests on $(git rev-parse --short HEAD)"

git submodule update --init --recursive || exit 1
uv sync --frozen --group dev --extra tabpfn || exit 1
uv sync --frozen --project envs/tabularargn || exit 1
uv sync --frozen --project envs/cdtd || exit 1

status=0
SYNTHDATA_REQUIRE_SIDECARS=1 uv run --with catboost==1.2.10 \
  pytest tests/integration/test_sidecar_generators.py -m slow -p no:cacheprovider || status=1
uv run --with catboost==1.2.10 \
  pytest tests/integration -m regression -p no:cacheprovider || status=1

echo "== $(date -Is) done, status $status"
exit "$status"
