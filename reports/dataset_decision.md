# Phase 1 - Dataset / target decision (DRAFT)

Status: **pending dataset access.** Fill in after reading TN 2367 and the SD 302 user guide.

## Audit findings (2026-10-21)
- CSFD-10k (HF jxZhang-hf/CSFD-10k): zip has 10,000 images (data/) + 10,000 edge maps (data_e/), BMP, ~263 KB each.
  **No MOS / label file, no dataset card, no license.** Cannot supervise training as downloaded.
- NIST SD 302: request form only (no direct download). Only SD 302h has expert ridge quality maps.
- => Chosen: SD 302h. CSFD-10k stays unused unless the authors supply MOS labels.

## SD 302h inspection (2026-10-21) - GATE PASSED
- 40,368 .lffs = 4 variants (original|enhanced x masked|unmasked) of 10,092 latents; participant id in folder name.
- Each .lffs holds Type-1/2 headers, Type-9 EFS annotations, Type-13 embedded PNG (302e NOT needed).
- Ridge quality map = field 9.308: integer grid, one row per RS, cell size 9.309 = 20 (to verify in TN 2367).
- Grid aspect matches image; overlay (outputs/figures/annotations.png) shows good visual alignment.
- Values seen up to 5; 0 = outside/background, higher = visually clearer ridges (confirm names in TN 2367).
- Heavy class imbalance (small sample: 0 ~54%, 1 ~37%, rest small) -> need class-weighted / ordinal loss.
- Parser: src/ebts.py ; viewer: src/viz_annotations.py

## Manifest (src/build_manifest.py, variant original/unmasked, seed 42)
- 10,092 files, all SHA-256 checksums match NIST's CSV. 200 participants.
- Excluded 71: 67 have no quality map (52 are from participant 00002649), 4 have a malformed duplicated 9.308 field.
- Usable 10,021. Subject-level split: train 6,400 (127 subj) / val 2,056 (41) / test 1,565 (31). Test not to be touched until final eval.
- Cell share by level 0..5: 46.5 / 29.5 / 16.5 / 5.1 / 1.7 / 0.7 %. Splits have similar distributions.
- Image size varies a lot: height 256-2142 (median 717), width 256-1974 (median 687).

## Decision
- Primary dataset: NIST SD 302h (expert ridge-quality maps)
- Primary target: regional quality map (NOT an image-level 0-100 score)
- OpenLQM: baseline only, run after the model is frozen. Never a training label.

## To verify from the real files (do not assume)
- [ ] Access request submitted / approved (date: ____)
- [ ] Subset + release version: ____
- [ ] Number of images: ____ ; annotation files: ____
- [ ] Ridge-quality map encoding (ordinal classes? values? geometry? missing-value code?): ____
- [ ] Image <-> annotation mapping is 1:1 and verified visually
- [ ] Subject / source grouping metadata available for leakage-safe splits: ____
- [ ] Usage / redistribution terms: ____

## Go / no-go (plan section 3.2)
Can a reliable regional target be built? ____
Is a defensible image-level scalar available? ____ (if no -> regional map only)

## Fallback
If access is delayed: reduce scope or use CSFD-10k (check terms). Do NOT train on unlabeled images.
