# Preserved experimental evidence

`experiment_09_20261103.zip` is the exact completed Experiment 9 deliverable.
Its root directory is `persona_shrinkage_pilot/`. The ZIP includes the exact
Experiment 8 deliverable at `provenance/source_v8.zip` and a second identical
copy inside the generation-time source snapshot. No model weights, Python
environment, caches, or partial model downloads are included.

| Evidence | SHA-256 | Files including manifest |
|---|---|---:|
| Experiment 9 ZIP | `65b7927bde72ec225acff1540bec2d88725ca810d1b58b82b79749b2d2a19909` | 177 |
| Embedded Experiment 8 ZIP | `0538ed94a04a2fa5f51195c56b66a4eabeb2ae56c0aa1cddb12787d0db8b77c8` | 175 |

The archive is 6,574,553 bytes. [index.json](index.json) records its identity and
the provenance of each browsable copy under [research/experiment_09](../research/experiment_09/).
Those copies match their original archive members byte for byte; their text and
recorded results have not been rewritten for this repository.

## Verify without a model or scientific packages

From the repository root, using Python 3.10 or later:

```bash
python scripts/verify_archive.py
```

The checker reads the archive without extracting it. It validates the ZIP's
SHA-256 and size against `index.json`, every manifest entry, complete archive
membership, both copies of the embedded Experiment 8 archive, and all browsable
copies. Nested duplicate archives are fully checked once and their second copy
is checked by the identical digest. It rejects unsafe paths, symlinks, duplicate
or case-colliding names, oversized payloads and unmanifested files.

`VERIFIED` means **byte and membership integrity**, not a new experiment or
semantic validation. The completed study's independent arithmetic audit is
preserved at `persona_shrinkage_pilot/audits/final.json`; that audit recomputed
counts, blocked cross-validation, selection, predictions and scores. These are
different checks. Local hashes and timestamps are not externally certified
timestamps or public preregistration.

## Inspect the original study

After a successful integrity check, extract a working copy to a new directory:

```bash
python -m zipfile -e archives/experiment_09_20261103.zip .local/archive-inspection
```

The original `README.md`, `setup/README.md`, model manifest and exact package
records explain the original CPU environment and the archived arithmetic audit.
Their machine-specific setup commands describe the executed study; use the new
repository's portable workflow for additional runs on a Mac or furnace.

To repeat the original final arithmetic audit, first restore the exact model
files required by its integrity checks, then use a new audit output filename in
the extracted working copy. Do not delete or overwrite completed locks, labels,
predictions or audit records to make a command run. A new inference run, backend,
precision setting or model revision is a separate run with its own provenance.

The archive contains every sampled token sequence, full visible completion,
semantic label and locked forecast. These executed records remain available
even where a different numerical stack produces different sampled text.

## Publication review

The repository preparation inspected both unique archived studies for credential
formats and signed URLs. The recorded scan found no matches for those checks;
see [publication_review.json](publication_review.json) for the reviewed hash and
scope. It is a bounded content check, not a guarantee that arbitrary future
archives are publishable. The experiments use authored fictional questions.

The repository's Apache-2.0 license covers its authored code and research
materials. External model weights are not redistributed here and keep their own
license. Cited papers are linked, not bundled or relicensed.
