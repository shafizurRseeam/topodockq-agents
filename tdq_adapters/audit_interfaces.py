"""Dataset audit: empty interface files in the published SinglePPD interface archive.

    python -m tdq_adapters.audit_interfaces

Read-only. Writes runs/audit-interfaces-<timestamp>/audit_interfaces_report.json
(via the run guard) and a copy of the findings to
artifacts/audits/singleppd_empty_interfaces.json.
"""
from __future__ import annotations

import json
import time

from . import data_audit
from .paths import PROJECT_ROOT, RUNS_DIR, ZENODO_DIR
from .provenance import sha256
from .runner import run_guarded

TAR = ZENODO_DIR / "singlePPD_interface_files.tar.gz"
OUT = PROJECT_ROOT / "artifacts" / "audits" / "singleppd_empty_interfaces.json"


def body(report, run_dir):
    report["archive_sha256"] = sha256(TAR)
    findings = data_audit.audit_interface_archive(TAR)
    report["findings"] = {k: v for k, v in findings.items() if k != "empty_keys"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"archive_sha256": report["archive_sha256"], **findings}, indent=1), encoding="utf-8")
    print(json.dumps(report["findings"], indent=1, default=str)[:3000])
    return {"archive_fully_parsed": not findings["unparsed_names"]}


if __name__ == "__main__":
    code, _ = run_guarded("audit_interfaces", RUNS_DIR / f"audit-interfaces-{time.strftime('%Y%m%d-%H%M%S')}", body)
    raise SystemExit(code)
