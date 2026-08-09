"""List which states each Tracking the Sun edition actually publishes.

`msa.ingest.tts.TTS_MIDWEST_STATES` hard-codes IL/MN/OH/WI because those are the
only Midwest states in the 2024 edition. This script is the check behind that
claim: it lists the OEDI bucket and prints, per edition, which Midwest states
are present.

Run::

    python -m scripts.verify_tts_states
"""

from __future__ import annotations

import json

from msa.config import MIDWEST_STATES, REPORTS_DIR
from msa.ingest.tts import _list_state_keys

EDITIONS = ("2018", "2019", "2020", "2021", "2022", "2023", "2024")


def main() -> None:
    out = {}
    for edition in EDITIONS:
        try:
            keys = _list_state_keys(edition)
        except Exception as exc:  # noqa: BLE001 - an absent edition is a result
            out[edition] = {"error": str(exc)}
            print(f"{edition}: unavailable ({exc})")
            continue
        states = sorted(keys)
        midwest = sorted(set(states) & set(MIDWEST_STATES))
        out[edition] = {"all_states": states, "midwest_states": midwest}
        print(f"{edition}: {len(states)} states, Midwest = {', '.join(midwest) or 'none'}")

    path = REPORTS_DIR / "tts_state_coverage.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
