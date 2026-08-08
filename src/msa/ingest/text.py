"""Shared scope-of-work text parsing used by every permit ingester.

Municipal permit registers describe the job in one free-text field. These
helpers turn that text into the handful of structured attributes the model
uses, and -- just as importantly -- screen out the non-PV work that the word
"solar" also appears in (solariums, solar water heating, solar-control film,
Solatube skylights).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

#: Candidate match. Deliberately **recall-first**: the final bare ``SOLAR``
#: alternative subsumes the specific phrases before it, so in practice this
#: matches "contains SOLAR, PHOTOVOLT or PV". Municipal scope-of-work text is
#: written by hundreds of different clerks and contractors ("Tesla Solar Roof",
#: "ballasted racking system for solar module", "COLLECTOR CAPACITY 6.16 KW"),
#: so an enumerated whitelist loses real installations. Precision is recovered
#: by ``NON_PV_RE`` below, which is the part that actually does the work.
#: The specific phrases are kept because they document the shapes seen in the
#: corpus and would still match if the bare alternative were ever tightened.
PV_RE = re.compile(
    r"SOLAR PANEL|SOLAR ARRAY|SOLAR PV|SOLAR SYSTEM|SOLAR ELECTRIC|SOLAR ENERGY"
    r"|SOLAR MODULE|SOLAR INSTALL|SOLAR ROOF|PHOTOVOLT|\bPV\b|ROOFTOP SOLAR"
    r"|SOLAR COLLECTOR|COLLECTOR CAPACITY|SOLAR"
)

#: Vetoes records where "solar" means something other than PV.
NON_PV_RE = re.compile(
    r"SOLARIUM"
    r"|SOLAR WATER HEAT|SOLAR HOT WATER|SOLAR THERMAL|SOLAR DOMESTIC"
    r"|SOLAR TUBE|SOLATUBE|SOLAR SCREEN|SOLAR SHADE|SOLAR FILM|SOLAR CONTROL"
    r"|SOLAR TINT|SOLAR LIGHT|SOLAR STREET"
)

_RE_KW = re.compile(r"([\d]+(?:\.[\d]+)?)\s*KW")
_RE_BATTERY = re.compile(r"BATTER|ENERGY STORAGE|\bESS\b|POWERWALL")
_RE_GROUND = re.compile(r"GROUND[- ]MOUNT|GROUND MOUNTED|POLE[- ]MOUNT")

_SUFFIX_RE = re.compile(
    r"\b(LLC|L L C|INC|INCORPORATED|CO|CORP|CORPORATION|LTD|LP|LLP|COMPANY|DBA)\b"
)


def extract_kw(text: pd.Series) -> pd.Series:
    """First plausible kW figure in the description."""
    return pd.to_numeric(text.str.extract(_RE_KW, expand=False), errors="coerce")


def has_battery(text: pd.Series) -> pd.Series:
    return text.str.contains(_RE_BATTERY, na=False)


def is_ground_mount(text: pd.Series) -> pd.Series:
    return text.str.contains(_RE_GROUND, na=False)


def normalize_contractor(name: object) -> object:
    """Upper-case, strip punctuation and corporate suffixes.

    ``ALL ENERGY SOLAR, INC.`` and ``All Energy Solar LLC`` collapse to the same
    key so contractor history accumulates across filings.
    """
    if not isinstance(name, str) or not name.strip():
        return np.nan
    s = re.sub(r"[^A-Z0-9 ]+", " ", name.upper())
    s = _SUFFIX_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or np.nan


def is_pv(text: pd.Series) -> pd.Series:
    """Boolean mask: genuine PV, not a solarium / solar tube / solar film."""
    t = text.fillna("").str.upper()
    return t.str.contains(PV_RE) & ~t.str.contains(NON_PV_RE)
