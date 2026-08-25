# Dependency and supply-chain notes

Everything here is reproducible with the commands given. Nothing is asserted
without one.

## Pinning

* `requirements.txt`: the direct dependencies, each pinned to an exact version.
* `requirements.lock.txt`: the full resolved environment (111 packages,
  every transitive dependency pinned). This is the environment the measured
  numbers came from and the one CI installs.
* `requirements-dev.txt`: linter and auditor only. Deliberately *not* in the
  lock file, so the reproduction environment is not contaminated by tooling
  that had nothing to do with producing the numbers.

Verify the lock is internally consistent:

```bash
.venv/bin/pip check          # -> "No broken requirements found."
```

Verify the installed environment matches the lock exactly:

```bash
diff <(sort requirements.lock.txt) <(.venv/bin/pip freeze | sort)
# the only expected difference is the editable install of this project itself
```

## Known vulnerabilities

Audited with `pip-audit`, which resolves each pin against the PyPI Advisory
Database and OSV:

```bash
pip install -r requirements-dev.txt
pip-audit -r requirements.lock.txt
```

As of the last run (2026-08-09) that reports **one** finding across all 111
pinned packages:

| package | pinned | advisory | fixed in |
|---|---|---|---|
| `cryptography` | 49.0.0 | [PYSEC-2026-3552](https://osv.dev/vulnerability/PYSEC-2026-3552) / CVE-2026-69247 (CVSS v4.0 8.2) | 50.0.0 |

**It is not fixed in this tree, and here is exactly why.** `cryptography` is a
transitive dependency, pulled in by `mlflow` and `google-auth`. The latest
release of MLflow, 3.15.1, declares `cryptography<50,>=43.0.0`. Installing the
fixed 50.0.0 therefore breaks the dependency graph:

```
$ pip install cryptography==50.0.0 && pip check
mlflow 3.15.1 has requirement cryptography<50,>=43.0.0, but you have cryptography 50.0.0.
```

There is no MLflow release that permits `cryptography>=50`, so the choice is
between a broken environment and this advisory. The advisory stays.

**Reachability.** The flaw is a Bleichenbacher-style padding oracle in PKCS#7
`EnvelopedData` decryption: decrypting a `RecipientInfo`'s `encryptedKey`
reveals the recovered key length through distinguishable errors and timing. It
is exploitable against a service that automatically decrypts PKCS#7 data
supplied by an attacker. This project never calls PKCS#7 decryption, not
directly and not through MLflow, whose use of `cryptography` here is limited
to the TLS and JWT paths that are not even exercised by a local SQLite tracking
store. Grep confirms it:

```bash
grep -rn "pkcs7\|EnvelopedData" src/ scripts/ tests/    # no matches
```

That is an argument that the finding is not exploitable *in this deployment*.
It is not an argument that the dependency is safe in general, and it is not a
reason to stop tracking it.

**CI behaviour.** The `audit` job runs `pip-audit` on every push with
`--ignore-vuln PYSEC-2026-3552`. A *new* advisory in any pinned package
therefore fails CI; this one documented finding does not. When MLflow relaxes
its pin, drop the ignore and bump `cryptography`.

## What the service exposes

* No secrets, credentials or connection strings are read at request time. The
  only environment variables the service consults are paths and feature flags
  (`MSA_SERVE_SEGMENT`, `MSA_MODELS_DIR`, `MSA_ALLOW_DEGRADED_START`,
  `MSA_MAX_BODY_BYTES`, `MSA_MAX_CONCURRENT_EXPLAIN`, `MSA_LOG_*`).
* Error responses carry a fixed code, a fixed human-readable string and a
  correlation id, never an exception message, a traceback or a filesystem
  path. `tests/test_serve_hardening.py::assert_no_leak` asserts this against
  every error path the suite can reach.
* Request bodies are capped (`MSA_MAX_BODY_BYTES`, default 4 MiB → 413),
  batches are capped at 1000 items, every string field is length-capped and
  every numeric field is range-capped. `Infinity` and `NaN` (which Python's
  JSON parser accepts) are rejected at validation.
* `SOCRATA_APP_TOKEN` is read by the *ingester*, never by the service, and is
  only ever sent as a request header to the Socrata domain being queried.

## Data

All source data is municipal, state or federal open data, fetched anonymously
over HTTPS. No credentials are required and none are stored. Cached raw
downloads live in `data/raw/` and are gitignored.
