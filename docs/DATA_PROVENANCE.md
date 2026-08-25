# Data provenance

Every row in this project comes from a public government open-data endpoint.
**No data is synthetic, simulated, resampled or imputed at the record level.**
(The only synthetic data in the repo lives in `tests/`, is named
`synthetic_canonical`, and is never used for training or for any number quoted
in the README.)

Re-download everything with:

```bash
.venv/bin/python -m msa.ingest.build_dataset --refresh
```

Raw payloads are cached under `data/raw/` (gitignored). `data/processed/dataset_summary.json`
is regenerated on every build and records the per-source counts below.

---

## 1. Permit registers: the source of the target variable

Twelve municipal building-permit registers across eight Midwest states. Each
publishes some subset of three milestone dates; the ingester maps whatever it
publishes onto the canonical `application_date` / `issue_date` /
`completion_date` triple.

| Jurisdiction | State | Platform | Dataset / layer | Milestones published | Solar identified by | Records |
|---|---|---|---|---|---|---|
| Chicago | IL | Socrata | [`ydr8-5enu` Building Permits](https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu) | application → issue | `work_type = 'Small-Scale Solar PV System'` (2023+) **or** `work_description` matches PV keywords | 12,873 |
| Kansas City | MO | Socrata | [`ntw8-aacc` Permits – CPD](https://data.kcmo.org/resource/ntw8-aacc) + [`w8jz-wjgn` City Issued Permits](https://data.kcmo.org/resource/w8jz-wjgn) | application → issue → completion | `description` / `project` text | 4,550 |
| Saint Paul | MN | ArcGIS | `Approved_Building_Permits/FeatureServer/0` | issue → final inspection | `SUB_TYPE IN ('Solar','Solar PV Systems')` **or** description text | 3,781 |
| Minneapolis | MN | ArcGIS | `CCS_Permits/FeatureServer/0` | issue → completion | `comments` text (no solar work-type code exists) | 3,443 |
| Cincinnati | OH | Socrata | [`thvx-5mem`](https://data.cincinnati-oh.gov/resource/thvx-5mem) joined to [`94uh-66xv`](https://data.cincinnati-oh.gov/resource/94uh-66xv) on `number_key` | application → issue → completion | `desc_of_work` text | 1,089 |
| Cleveland | OH | ArcGIS | `Building_Permits/FeatureServer/0` | file → issue | `WORK_DESCRIPTION` text | 1,066 |
| Naperville | IL | ArcGIS | `Building_Permits_View/FeatureServer/0` | apply → issue → finalize | `PERMITWORKCLASS = 'SOLAR/PHOTOVOLTAIC'` | 850 |
| Detroit | MI | ArcGIS | `bseed_building_permits/FeatureServer/0` | submitted → issued | `work_description` text | 697 |
| Golden Valley | MN | ArcGIS | `Golden_Misc/FeatureServer/586` + `/587` | applied → issued → final | `Sub_Type`/`Work_Type`/`Description` | 670 |
| Ames | IA | ArcGIS | `coa_Inspections_BuildingPermits_V2/MapServer/2` | application → issue → finaled | `DESCRIPTION` text | 262 |
| Overland Park | KS | ArcGIS | `Building_Permits/FeatureServer/0` | apply → issue → finalized | `WorkClass = 'Solar Panel'` | 128 |
| Bismarck | ND | ArcGIS | `Map1/FeatureServer/0` | applied → issued → finaled | `DESCRIPTION` text | 16 |

**Total canonical records: 29,425 solar PV permits.**

### Non-PV screening

"Solar" appears in permit text for work that is not photovoltaic. Every
text-matched source runs the shared veto list in `msa/ingest/text.py`:
solariums, solar water/thermal heating, Solatube skylights, solar screens /
shades / film / window tint, solar street lighting. `tests/test_ingest.py`
locks the classifier's behaviour on worked examples from the real corpus.

### How the target is derived

`timeline_days = (end_milestone - start_milestone).days`, with the pair chosen
by the `segment` setting:

| Segment | Start | End | Meaning | Usable rows |
|---|---|---|---|---|
| `permit_to_completion` | `issue_date` | `completion_date` | field installation window: approved to build → final inspection passed / permit closed | 12,108 |
| `application_to_issue` | `application_date` | `issue_date` | permit review window | 21,884 |
| `application_to_completion` | `application_date` | `completion_date` | whole chain (only where a register publishes all three) | 5,960 |

Two filters are applied before modelling, both in `msa/features/build.py::clean`:

1. **Horizon.** Durations outside `[0, horizon]` are treated as out of scope
   rather than squashed into the target. Measured coverage
   (`python -m scripts.report_horizon_coverage`): 365 days covers **93.0%** of
   `permit_to_completion` rows, 120 days covers **98.6%** of
   `application_to_issue` rows.
2. **Right-censoring guard.** A project whose clock started fewer than
   `horizon` days before the snapshot can only appear in an open-data extract
   if it *already* finished, so the recent tail is biased short. Rows whose
   start date is after `max(end_date) − horizon` for that source are dropped.
   This is why the modelled row count (e.g. 10,162 for `permit_to_completion`)
   is lower than the raw usable count (12,108).

---

## 2. Utility records: LBNL *Tracking the Sun* (context features only)

* **Source**: Lawrence Berkeley National Laboratory, *Tracking the Sun*, the
  public data file of U.S. distributed PV systems compiled from state incentive
  programmes and **utility interconnection registries**.
* **Access**: DOE/NREL Open Energy Data Initiative public S3 bucket,
  `https://oedi-data-lake.s3.amazonaws.com/tracking-the-sun/2024/state=<XX>/`,
  anonymous HTTPS, Hive-partitioned by edition and state.
* **Midwest coverage in the 2024 edition**: IL, MN, OH, WI only, **98,275
  systems**. (IN, IA, MI, NE, ND, SD are not published at all; KS and MO appear
  only in the 2018-2020 editions. Verified by listing the bucket:
  `python -m scripts.verify_tts_states`, output in
  `reports/tts_state_coverage.json`.)

**Critical limitation, verified not assumed:** the public Tracking the Sun file
contains exactly **one** date column, `installation_date`. It therefore cannot
supply a duration and is **not** a source of the target. It is used only as a
market-context layer: as-of joins that attach, for each permit,
the cumulative installs / mean system size / median $-per-watt / third-party-owned
share in that ZIP and the trailing-365-day install volume in that state, all
computed over TTS systems installed **strictly before** the permit's start date.

---

## 3. Sources evaluated and rejected

Recorded so the search is auditable rather than repeated.

| Source | Why rejected |
|---|---|
| **Illinois Shines / Adjustable Block Program** (`illinoisshines.com/project-application-reports/`), ~117,000 Illinois projects, joinable on `Application ID`, with several real date pairs | Measures **programme administration** (Part I submission → ICC approval; Part II submission → verification), not permitting or construction. Its `Online Date` runs *backwards* relative to approval (median −104 days) because vendors routinely energise first and file after, so it cannot anchor an installation timeline. Excluded to keep one coherent target definition rather than to avoid the work |
| Cincinnati `uhjb-xac9` (headline Building Permits) | Its `description` column is a 16-value category label, not scope text; `like '%SOLAR%'` returns 0. Full-text hits are the street "SOLAR VISTA PL" and contractor names. Solar had to be identified in `thvx-5mem` instead |
| Sioux Falls SD | 158,752 permits with a textbook `APPLYDATE`/`ISSUEDATE`/`FINALIZEDATE` schema and contractor names, but **no description field at all** and no solar work class, so solar is unidentifiable |
| Columbus OH | 677,179 permits, no free-text field, no solar category; only proxy is contractor business name |
| Cook County IL Assessor Permits | 26,074 solar rows but `estimated_date_of_completion` is a *text* column populated on 414 of 719,506 rows, one date in practice |
| Michigan MPSC Distributed Generation, Ohio PUCO DER, Illinois Solar for All, Iowa IUC, Missouri PSC, Kansas CC | One date, or aggregate-only reporting mandated by regulation |
| NREL SolarTRACE | Exactly the permit → inspection → interconnection durations wanted, but published as **AHJ × utility × year medians**, not row level. `data.nrel.gov` was also unreachable from this environment |
| MISO / SPP interconnection queues | Utility-scale only (372 + 93 Midwest solar projects); `inService` is a *planned* date |
| Madison WI, Milwaukee WI, Ann Arbor MI, Grand Rapids MI, Indianapolis, Des Moines, Lincoln NE, Omaha, Fargo, Toledo, Dayton, Akron, Wichita, Topeka, St. Louis, and ~30 others | No public permit API, no solar identification, a single date, or a degenerate duration (St. Louis: median 1 day) |

---

## Licensing

All sources are municipal, state or federal open data published without
restriction on reuse. Tracking the Sun is public data, free to use with
attribution to LBNL.
