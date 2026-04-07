# frankentorch-xsp import-path investigation report

## Scope

This report captures the results of the `frankentorch-xsp` investigation into the recurring
`.beads/beads.db` integrity warning:

`PRAGMA integrity_check` -> `Page N: never used`

The question was which write path reintroduced the anomaly after a clean `VACUUM INTO` rewrite.

## Inputs

- clean baseline DB:
  - `.beads/beads.db.vacuum_c84_20260407T013244Z`
- current tracker JSONL:
  - `.beads/issues.jsonl`
- archived probe artifacts:
  - `artifacts/beads_recovery/xsp_probe_20260407T025948Z/results.json`
  - `artifacts/beads_recovery/xsp_import_focus_20260407T030054Z/results.json`
  - `artifacts/beads_recovery/xsp_sequence_20260407T030136Z/results.json`
  - `artifacts/beads_recovery/xsp_repeat_import_20260407T030112Z/results.json`

## Key findings

1. The clean vacuumed DB is genuinely clean.
   - baseline page count: `89`
   - baseline freelist count: `0`
   - `integrity_check`: `ok`

2. The corruption is not caused by generic read-only access or the `br doctor` write probe.
   - preserved integrity:
     - `br info --json`
     - `br doctor`
     - `br doctor --no-auto-flush --no-auto-import`
     - `br sync --status --json`
     - `br list --json --no-auto-import`
     - `br ready --json --no-auto-import`

3. The corruption is reintroduced by the import path.
   - explicit trigger:
     - `br sync --import-only --json`
   - implicit triggers through auto-import:
     - `br list --json`
     - `br ready --json`
     - `br show frankentorch-xsp --json`
     - `br update frankentorch-xsp --status ... --json`

4. The shared signature is `blocked_issues_cache` object churn.
   - clean baseline blocked-cache objects:
     - table rootpage `19`
     - PK autoindex rootpage `20`
     - `idx_blocked_cache_blocked_at` rootpage `56`
   - after import:
     - table rootpage `96`
     - PK autoindex rootpage `95`
     - `idx_blocked_cache_blocked_at` rootpage `97`
   - post-import integrity warning:
     - `Page 19: never used`
     - `Page 20: never used`
     - `Page 56: never used`

5. Inference:
   - The import path rebuilds or recreates `blocked_issues_cache`.
   - The new table/index objects land on fresh pages.
   - The old blocked-cache pages are not returned to the freelist correctly.
   - This leaves orphaned pages that `integrity_check` reports as `never used`.

## Reproduction summary

### Probe matrix

Starting from the clean vacuumed DB plus current JSONL:

- safe:
  - `br info --json`
  - `br doctor`
  - `br sync --status --json`
- unsafe:
  - `br sync --import-only --json`
  - any command that auto-imports before operating

### Minimal reproducer

1. Start from `.beads/beads.db.vacuum_c84_20260407T013244Z`
2. Pair it with the current `.beads/issues.jsonl`
3. Run `br sync --import-only --json`
4. Run `PRAGMA integrity_check`

Observed result:

- page count grows from `89` to `97`
- freelist count becomes `4`
- integrity check reports:
  - `Page 19: never used`
  - `Page 20: never used`
  - `Page 56: never used`

## Operational guidance

Until the upstream bug is fixed:

- treat the import path as the corrupting operation
- use `--no-auto-import` when inspecting a known-good vacuumed recovery DB
- prefer `br doctor` or `br info` for non-mutating diagnostics on recovered DBs
- assume any command that auto-imports can reintroduce the warning

## Remaining work

This investigation isolated the failing class of operations, but it did not fix the upstream
writer/import implementation in `beads_rust` / frankensqlite. A follow-up bead should track:

- upstream escalation / fix
- a local safe-workflow note for future tracker recovery sessions
