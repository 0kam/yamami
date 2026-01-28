# Tasks: Yamami Phenology Pipeline

**Input**: Design documents from `/specs/001-phenology-pipeline/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓

**Tests**: TDD is MANDATORY per Constitution Principle IV. Tests MUST be written and FAIL before implementation.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project directory structure per plan.md (src/yamami/, tests/, docs/)
- [ ] T002 Create pyproject.toml with dependencies (numpy, pandas, opencv-python, scipy, pillow, tqdm, segment-anything, image-matching-models)
- [ ] T003 [P] Create src/yamami/__init__.py with package exports, version, and public API
- [ ] T004 [P] Create .readthedocs.yaml configuration
- [ ] T005 [P] Create tests/conftest.py with shared pytest fixtures
- [ ] T006 [P] Create docs/conf.py Sphinx configuration with RTD theme and i18n

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

### Tests for Foundational ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T007 [P] Unit test for logging setup in tests/unit/test_logging.py
- [ ] T008 [P] Unit test for filename parser in tests/unit/test_parser.py

### Implementation for Foundational

- [ ] T009 [P] Implement logging setup with structured formatter in src/yamami/logging.py (FR-030)
- [ ] T010 [P] Implement filename parser for `{site}_{azimuth}_{camera}_{band}_{yyyymmdd}_{hhmm}` pattern in src/yamami/parser.py (FR-002)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Image Collection and Profiling (Priority: P1) 🎯 MVP

**Goal**: Catalog time-lapse images with metadata and compute quality statistics to identify unusable images

**Independent Test**: Test with a sample image directory to produce a complete profile DataFrame

### Tests for User Story 1 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T011 [P] [US1] Contract test for ingest() return type (DataFrame) in tests/contract/test_ingest_profile.py
- [ ] T012 [P] [US1] Contract test for profile() return type (DataFrame) in tests/contract/test_ingest_profile.py
- [ ] T013 [P] [US1] Unit test for directory scanner in tests/unit/test_ingest.py
- [ ] T014 [P] [US1] Unit test for luminance statistics in tests/unit/test_ingest.py
- [ ] T015 [P] [US1] Unit test for EXIF extraction in tests/unit/test_ingest.py

### Implementation for User Story 1

- [ ] T016 [US1] Implement ingest() to find JPEG/TIFF files and return DataFrame in src/yamami/ingest.py (FR-001)
- [ ] T017 [US1] Implement compute_luminance_stats() for mean, std, percentiles in src/yamami/ingest.py (FR-003)
- [ ] T018 [US1] Implement extract_exif() using Pillow in src/yamami/ingest.py (FR-004)
- [ ] T019 [US1] Implement profile() to return DataFrame with all metadata in src/yamami/ingest.py (FR-003)
- [ ] T020 [US1] Export ingest, profile in src/yamami/__init__.py
- [ ] T021 [US1] Integration test for ingest→profile workflow in tests/integration/test_pipeline.py

**Checkpoint**: User Story 1 complete - can catalog and profile images independently

---

## Phase 4: User Story 2 - Quality Control and Image Alignment (Priority: P2)

**Goal**: Identify obscured images (fog, cloud, darkness), detect camera shifts, and align to reference

**Independent Test**: Test with images containing known fog/cloud and camera shift events

### Tests for User Story 2 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T022 [P] [US2] Contract test for segment() return types in tests/contract/test_segment_qc.py
- [ ] T023 [P] [US2] Contract test for pre_qc() return type in tests/contract/test_segment_qc.py
- [ ] T024 [P] [US2] Contract test for align() return types in tests/contract/test_align_aoi.py
- [ ] T025 [P] [US2] Unit test for SAM3 segmentation in tests/unit/test_segment.py
- [ ] T026 [P] [US2] Unit test for QC thresholding in tests/unit/test_qc.py
- [ ] T027 [P] [US2] Unit test for phase correlation change detection in tests/unit/test_align.py
- [ ] T028 [P] [US2] Unit test for homography estimation in tests/unit/test_align.py

### Implementation for User Story 2

#### Segmentation (FR-005~007)
- [ ] T029 [US2] Implement segment() with SAM3 inference and text prompts in src/yamami/segment.py (FR-005)
- [ ] T030 [US2] Implement mask rescaling to original dimensions in src/yamami/segment.py (FR-006)
- [ ] T031 [US2] Implement mask output as DataFrame + PNG files in src/yamami/segment.py (FR-007)

#### Quality Control (FR-008~010)
- [ ] T032 [US2] Implement pre_qc() with brightness/fog/cloud threshold checks in src/yamami/qc.py (FR-008)
- [ ] T033 [US2] Implement reason code generation (TOO_DARK, HIGH_FOG, etc.) in src/yamami/qc.py (FR-009)
- [ ] T034 [US2] Implement qc() with AOI consideration in src/yamami/qc.py (FR-010)

#### Alignment (FR-011~017)
- [ ] T035 [US2] Implement phase_correlation_change_detect() for shift detection in src/yamami/align.py (FR-011)
- [ ] T036 [US2] Implement segment_timeseries() to create stable intervals in src/yamami/align.py (FR-012)
- [ ] T037 [US2] Implement select_anchor_images() based on quality scores in src/yamami/align.py (FR-013)
- [ ] T038 [US2] Implement match_to_reference() using imm library in src/yamami/align.py (FR-014, FR-016)
- [ ] T039 [US2] Implement apply_warp() to transform images in src/yamami/align.py (FR-015)
- [ ] T040 [US2] Implement align() main function returning segments, warp_params in src/yamami/align.py (FR-017)

#### Skyline and AOI (FR-018~020)
- [ ] T041 [US2] Implement aoi() to extract skyline and create AOI mask in src/yamami/aoi.py (FR-018, FR-019)

- [ ] T042 [US2] Export segment, pre_qc, qc, align, aoi in src/yamami/__init__.py
- [ ] T043 [US2] Integration test for segment→pre_qc→align→aoi→qc workflow in tests/integration/test_pipeline.py

**Checkpoint**: User Story 2 complete - can filter and align images independently

---

## Phase 5: User Story 3 - Snowmelt Date Estimation (Priority: P3)

**Goal**: Determine snowmelt day-of-year at each pixel location across the mountain slope

**Independent Test**: Test with aligned QC-passed images from a complete snow season

### Tests for User Story 3 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T044 [P] [US3] Contract test for snow() return types in tests/contract/test_snow.py
- [ ] T045 [P] [US3] Contract test for snowmelt() return type in tests/contract/test_snow.py
- [ ] T046 [P] [US3] Unit test for snow threshold classification in tests/unit/test_snow.py
- [ ] T047 [P] [US3] Unit test for snowmelt DOY estimation in tests/unit/test_snow.py
- [ ] T048 [P] [US3] Unit test for rock exclusion in tests/unit/test_snow.py

### Implementation for User Story 3

- [ ] T049 [US3] Implement snow() with Otsu thresholding in src/yamami/snow.py (FR-021)
- [ ] T050 [US3] Implement detect_high_reflectance_rock() for exclusion in src/yamami/snow.py (FR-023)
- [ ] T051 [US3] Implement snowmelt() with multi-stage DOY analysis in src/yamami/snow.py (FR-022)
- [ ] T052 [US3] Export snow, snowmelt in src/yamami/__init__.py
- [ ] T053 [US3] Integration test for snow→snowmelt workflow in tests/integration/test_pipeline.py

**Checkpoint**: User Story 3 complete - can produce snowmelt DOY maps independently

---

## Phase 6: User Story 4 - Vegetation Greenup Analysis (Priority: P4)

**Goal**: Track seasonal vegetation greenup and extract green-up/green-down dates

**Independent Test**: Test with a complete growing season of aligned, QC-passed images

### Tests for User Story 4 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T054 [P] [US4] Contract test for gr() return types in tests/contract/test_phenology.py
- [ ] T055 [P] [US4] Contract test for phenology() return type in tests/contract/test_phenology.py
- [ ] T056 [P] [US4] Unit test for GR calculation in tests/unit/test_phenology.py
- [ ] T057 [P] [US4] Unit test for daily max aggregation in tests/unit/test_phenology.py
- [ ] T058 [P] [US4] Unit test for double-sigmoid fitting in tests/unit/test_phenology.py
- [ ] T059 [P] [US4] Unit test for inflection point extraction in tests/unit/test_phenology.py

### Implementation for User Story 4

- [ ] T060 [US4] Implement gr() with GR = G/(R+G+B) and daily max aggregation in src/yamami/phenology.py (FR-024, FR-025)
- [ ] T061 [US4] Implement fit_double_sigmoid() using scipy.optimize in src/yamami/phenology.py (FR-026)
- [ ] T062 [US4] Implement phenology() to extract green-up/green-down dates in src/yamami/phenology.py (FR-027)
- [ ] T063 [US4] Export gr, phenology in src/yamami/__init__.py
- [ ] T064 [US4] Integration test for gr→phenology workflow in tests/integration/test_pipeline.py

**Checkpoint**: User Story 4 complete - can produce phenology date maps independently

---

## Phase 7: User Story 5 - Results Export and Documentation (Priority: P5)

**Goal**: Export all results in standard formats with complete processing logs

**Independent Test**: Test with any completed analysis run

### Tests for User Story 5 ⚠️

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T065 [P] [US5] Contract test for viz() return type in tests/contract/test_export.py
- [ ] T066 [P] [US5] Contract test for export() behavior in tests/contract/test_export.py
- [ ] T067 [P] [US5] Unit test for DOY colormap visualization in tests/unit/test_viz.py
- [ ] T068 [P] [US5] Unit test for export directory organization in tests/unit/test_export.py

### Implementation for User Story 5

- [ ] T069 [US5] Implement viz() with DOY colormap and legends in src/yamami/viz.py (FR-028)
- [ ] T070 [US5] Implement export() to organize outputs and write processing log in src/yamami/export.py (FR-029, FR-030)
- [ ] T071 [US5] Export viz, export in src/yamami/__init__.py
- [ ] T072 [US5] Integration test for viz→export workflow in tests/integration/test_pipeline.py

**Checkpoint**: User Story 5 complete - can export and document results independently

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, packaging finalization, and quality assurance

### Documentation (FR-036~040)

- [ ] T073 [P] Create docs/index.rst with documentation structure
- [ ] T074 [P] Create docs/installation.md with all dependency options (FR-040)
- [ ] T075 [P] Copy quickstart.md to docs/quickstart.md (FR-039)
- [ ] T076 [P] Create docs/api/ with autodoc configuration (FR-038)
- [ ] T077 Setup gettext and create docs/locale/ja/ for Japanese translation (FR-037)

### Final Testing

- [ ] T078 Full end-to-end integration test in tests/integration/test_pipeline.py
- [ ] T079 Validate quickstart.md workflow with test data
- [ ] T080 Verify pip install yamami works in clean virtualenv (FR-031)
- [ ] T081 Verify ReadTheDocs build succeeds (FR-036)

### Coverage and Quality

- [ ] T082 Run pytest-cov and verify ≥80% coverage for core modules
- [ ] T083 Run pytest-cov and verify ≥60% coverage for utilities
- [ ] T084 Code review and cleanup

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - US1 (P1): Can proceed independently after Foundational
  - US2 (P2): Can proceed independently after Foundational
  - US3 (P3): Depends on US2 (needs aligned images and AOI)
  - US4 (P4): Depends on US2 (needs aligned images and AOI)
  - US5 (P5): Can proceed independently after Foundational
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

```
Phase 2 (Foundational)
    ↓
    ├── US1 (Ingest/Profile) → independent
    ├── US2 (QC/Align) → independent
    │   ↓
    │   ├── US3 (Snowmelt) → needs aligned images + AOI
    │   └── US4 (Greenup) → needs aligned images + AOI
    └── US5 (Export/Viz) → independent (can work with any outputs)
         ↓
Phase 8 (Polish)
```

### Within Each User Story

1. Tests MUST be written and FAIL before implementation (TDD)
2. Core functions before output writers
3. Export functions to __init__.py after implementation
4. Integration test last to verify workflow

### Parallel Opportunities

- All Phase 1 tasks can run in parallel
- All Phase 2 tests (T007-T008) can run in parallel
- All Phase 2 implementations (T009-T010) can run in parallel
- Within each user story, all tests marked [P] can run in parallel
- US1, US2, US5 can be developed in parallel after Phase 2
- US3, US4 can be developed in parallel after US2

---

## Implementation Strategy

### MVP First (User Story 1 + 2)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 (Ingest/Profile)
4. **VALIDATE**: Test ingest→profile independently
5. Complete Phase 4: User Story 2 (QC/Align)
6. **VALIDATE**: Test full pipeline through alignment

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 → Can catalog and profile images
3. Add US2 → Can filter and align images
4. Add US3 → Can produce snowmelt maps
5. Add US4 → Can produce phenology maps
6. Add US5 → Can export and visualize all results
7. Polish → Production-ready

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- TDD is MANDATORY: Write failing tests before implementation
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Coverage targets: ≥80% core modules, ≥60% utilities (Constitution Principle IV)
- No CLI: All functionality exposed as Python functions via __init__.py
- No config files: All parameters passed as function arguments
