#!/usr/bin/env bash
# Run every test of every module with no model download and no GPU, and
# write what was SKIPPED into the job summary. The program is
# scripts/modelfree, Go so that the same one runs on the Windows runner; its
# doc comment says what is run, what is left out and why.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec go run ./scripts/modelfree
