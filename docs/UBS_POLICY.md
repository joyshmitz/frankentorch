# UBS Security Scanner Policy

FrankenTorch uses [UBS (Ultimate Bug Scanner)](https://github.com/nightowlqa/ubs) for
security scanning of Rust code.

## Pre-commit Hook

The pre-commit hook runs `ubs --staged --only=rust` on staged files.

### Installation

```bash
cp hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Large File Handling

`crates/ft-api/src/lib.rs` is ~86K lines and cannot complete a full UBS scan within
the standard 120-second timeout. The pre-commit hook handles this by:

1. Detecting staged files >50K lines
2. Extending timeout to 300 seconds for large files
3. Providing `UBS_SKIP=1` escape hatch if needed

### Manual Scanning

For full project scans (CI or manual):

```bash
# Full project (may take several minutes)
ubs --only=rust

# Staged files only (recommended for pre-commit)
ubs --staged --only=rust

# Specific file with extended timeout
timeout 300 ubs --only=rust crates/ft-api/src/lib.rs
```

### Bypassing for Large Files

If the hook times out on large files and you need to commit urgently:

```bash
UBS_SKIP=1 git commit -m "message"
```

**Important**: Always run a full UBS scan on large files before merging to main.

## CI Integration

CI pipelines should run UBS without timeout constraints:

```bash
ubs --only=rust --format=sarif --ci
```
