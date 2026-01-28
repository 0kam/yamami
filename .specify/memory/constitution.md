<!--
============================================================================
SYNC IMPACT REPORT
============================================================================
Version change: 0.0.0 → 1.0.0 (MAJOR: Initial constitution adoption)
Modified principles: N/A (initial creation)
Added sections:
  - Core Principles (5 principles)
  - Development Workflow
  - Governance
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md: ✅ Compatible (Constitution Check section exists)
  - .specify/templates/spec-template.md: ✅ Compatible (requirements format aligns)
  - .specify/templates/tasks-template.md: ✅ Compatible (TDD workflow referenced)
Follow-up TODOs: None
============================================================================
-->

# yamami Constitution

## Core Principles

### I. Reproducibility First

All processing pipelines MUST be fully reproducible. Given the same input images and configuration, the system MUST produce identical outputs.

- Configuration files (YAML) MUST capture all parameters affecting processing
- Random seeds MUST be fixed and logged when stochastic operations are used
- Processing logs MUST record input file hashes, parameter values, and software versions
- Intermediate outputs MUST be preserved in documented formats (CSV for tables, PNG/TIFF for images)

**Rationale**: Scientific research demands reproducible results. Phenology and snowmelt analyses inform ecological studies where data provenance is critical.

### II. Pipeline Modularity

Each processing stage MUST be an independent, composable unit with clear input/output contracts.

- Stages MUST accept standard inputs (image paths, DataFrames) and produce standard outputs
- Stages MUST NOT have hidden dependencies on other stages' internal state
- Each module MUST expose a clear public API (functions with typed arguments)
- Parameters MUST be passed as function arguments, not global state

**Rationale**: Modular pipelines enable selective reprocessing, easier debugging, and gradual enhancement without system-wide changes.

### III. Simplicity Over Cleverness

Code MUST be straightforward and readable. Avoid premature optimization and over-engineering.

- Prefer explicit code over implicit "magic"
- Use standard library solutions before third-party dependencies
- Keep functions focused: single responsibility, under 50 lines preferred
- Comments explain "why", not "what" (code should be self-documenting)
- YAGNI: Do not implement features until actually needed

**Rationale**: Research code often outlives its original authors. Simple code is maintainable code.

### IV. Test-Driven Development (NON-NEGOTIABLE)

Tests MUST be written before implementation. Red-Green-Refactor cycle is strictly enforced.

- Write failing test → Get user approval → Implement → Test passes → Refactor
- Contract tests for inter-stage interfaces
- Integration tests for end-to-end pipeline flows
- Test data MUST be versioned and documented
- Coverage targets: Core processing logic ≥ 80%, Utilities ≥ 60%

**Rationale**: TDD ensures correctness from the start and prevents regression when adding features or fixing bugs.

### V. Traceability and Logging

Every processing decision MUST be traceable from output back to input.

- All outputs MUST include metadata linking to source images and parameters
- Status columns (e.g., `align_status`, `qc_status`) MUST use standardized codes
- Warnings and errors MUST be logged with context (file path, timestamp, parameter values)
- Failed processing MUST produce diagnostic artifacts, not silent failures

**Rationale**: When analysis results are questioned, the system must support forensic investigation of how results were derived.

## Development Workflow

### Code Review Requirements

- All changes MUST be reviewed before merging to main branch
- Reviewers MUST verify compliance with Constitution principles
- Changes affecting core processing logic require at least one domain expert review

### Branch Strategy

- `main`: Stable, tested, production-ready code
- Feature branches: `feature/[description]` for new capabilities
- Bugfix branches: `fix/[description]` for corrections

### Commit Standards

- Commits MUST be atomic and focused on a single change
- Commit messages MUST follow conventional format: `type: description`
  - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
- Each commit MUST pass all tests before being pushed

### Pull Request Process

1. Create PR with clear description of changes
2. Link to relevant issues or specifications
3. Ensure all CI checks pass
4. Obtain required approvals
5. Squash and merge to main

## Governance

### Authority

This Constitution supersedes all other development practices and conventions. In case of conflict, Constitution principles prevail.

### Amendment Process

1. Propose amendment with rationale
2. Document impact on existing code and practices
3. Obtain team consensus
4. Update Constitution with version increment
5. Update dependent templates and documentation

### Versioning Policy

- **MAJOR**: Backward-incompatible principle changes or removals
- **MINOR**: New principles or material guidance additions
- **PATCH**: Clarifications, typo fixes, non-semantic refinements

### Compliance Review

- All PRs MUST verify compliance with Constitution principles
- Quarterly audits of codebase against Constitution standards
- Complexity MUST be justified against Simplicity principle

### Guidance Reference

For runtime development guidance specific to implementation details, refer to:
- `YAMAMI_PIPELINE_DESIGN.md` for pipeline architecture
- `.specify/templates/` for specification and planning templates

**Version**: 1.0.0 | **Ratified**: 2026-01-27 | **Last Amended**: 2026-01-27
