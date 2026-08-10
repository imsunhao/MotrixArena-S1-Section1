#!/usr/bin/env python3
"""Restore normalization state from a stable checkpoint.

PPO fine-tuning updates running observation statistics even when policy
weights barely move. For a fragile locomotion warm start this can change the
effective policy much more than the optimizer update itself. This utility
keeps a candidate's learned model parameters while restoring the state/value
preprocessors from a known stable checkpoint.
"""

import hashlib
import pickle
from pathlib import Path

from absl import app, flags


_REFERENCE = flags.DEFINE_string(
    "reference", None, "Stable checkpoint providing normalization state"
)
_CANDIDATE = flags.DEFINE_string(
    "candidate", None, "Fine-tuned checkpoint providing model parameters"
)
_OUTPUT = flags.DEFINE_string("output", None, "Output checkpoint path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    del argv
    if not _REFERENCE.value or not _CANDIDATE.value or not _OUTPUT.value:
        raise app.UsageError("--reference, --candidate and --output are required")

    reference_path = Path(_REFERENCE.value)
    candidate_path = Path(_CANDIDATE.value)
    output_path = Path(_OUTPUT.value)
    if output_path in (reference_path, candidate_path):
        raise app.UsageError("--output must not overwrite an input checkpoint")

    with reference_path.open("rb") as stream:
        reference = pickle.load(stream)
    with candidate_path.open("rb") as stream:
        candidate = pickle.load(stream)

    required = ("state_preprocessor", "value_preprocessor")
    for key in required:
        if key not in reference or key not in candidate:
            raise KeyError(f"checkpoint is missing required key: {key}")
        candidate[key] = reference[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(candidate, stream)

    print(f"reference_sha256={_sha256(reference_path)}")
    print(f"candidate_sha256={_sha256(candidate_path)}")
    print(f"output_sha256={_sha256(output_path)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    app.run(main)
