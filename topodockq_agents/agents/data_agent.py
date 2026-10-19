"""DataAgent: turn the user's input into validated structures and their protein-peptide interfaces."""
from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from ..config import AgentConfig
from ..state import Keys
from ..tools import data_tools as dt
from .base import COMMON_RULES, BaseAgent

DOMAIN = """\
Context: TopoDockQ Agents scores AlphaFold candidate structures of peptide-protein complexes with the
published TopoDockQ method (Dai, Wang & Zhang, Commun. Chem. 2025). For each candidate it predicts DockQ
(p-DockQ, 0 to 1) from topological features of the protein-peptide interface, then picks the best candidate
per complex. Structures have the protein as chain A and the peptide as chain B. The feature generator writes
2,754 raw features per structure and the published model uses 2,646 of them; the other 108 are constant in
the training data and dropped by the authors' preprocessing. That difference is expected, not a problem.
"""


class DataAgent(BaseAgent):
    name = "DataAgent"
    role = ("Discovers the input structures, validates them with the fixed rules, and extracts each "
            "protein-peptide interface with the authors' script; produces data.structures, data.interfaces, data.report.")

    @property
    def cfg(self) -> AgentConfig:
        return self.settings.data_agent

    def system_prompt(self) -> str:
        return f"""You are DataAgent, the first stage of TopoDockQ Agents.

{DOMAIN}
Your job: turn the user's input into a validated list of structures and extract each structure's
protein-peptide interface (all residues within 10 A of the other chain) with the authors' script.

Workflow:
1. build_structure_manifest with the input source and the selection the task asks for. Accepted inputs:
   a folder with one sub-folder per complex holding AlphaFold models ranked_<N>.pdb / ranked_<N>_sp.pdb,
   one such complex folder on its own, a folder of PDB files, or a manifest .csv (pdb_id, af_model_id,
   input_pdb). Use complexes / models /
   max_complexes / max_models_per_complex only to express what the task asks for; with no narrowing in the
   task, take everything. Keep protein_chain="A", peptide_chain="B" unless the task states other chains
   (the authors' feature code reads only chains A and B, so other chain IDs are excluded by rule E_CHAIN_IDS).
   Set chains_confirmed=True only when the request explicitly states that chain A is the protein and chain B
   the peptide (or a manifest lists the chains). Unconfirmed roles that look implausible get the warning
   W_CHAIN_ROLE_SUSPICIOUS; report those structures, never drop them.
2. check_structures: a preflight with the fixed rules. If nearly every structure fails the same rule, the
   input or the chain assignment is probably wrong; correct the manifest arguments only if the task supports
   it. Otherwise continue: exclusions are applied and recorded by the tools, not by you.
3. prepare_interfaces with workers={self.settings.workers}. It validates every structure again, runs the
   authors' extract_interface.py, and records each exclusion with its rule code.
4. write_data_report with short notes: what was selected, how many structures are ready, which were
   excluded and under which rule codes.

Outputs you own: {Keys.STRUCTURES}, {Keys.INTERFACES}, {Keys.DATA_REPORT}.

{COMMON_RULES}"""

    def user_message(self, task: str) -> str:
        source = self.settings.input_source or "not given in the settings; use the source named in the task"
        return f"Input source provided by the user: {source}\n\n" + super().user_message(task)

    def build_tools(self) -> list[Any]:
        agent = self

        @beta_tool
        def build_structure_manifest(source: str, layout: str = "auto", complexes: list[str] | None = None,
                                     max_complexes: int | None = None, models: list[int] | None = None,
                                     max_models_per_complex: int | None = None, protein_chain: str = "A",
                                     peptide_chain: str = "B", chains_confirmed: bool = False) -> str:
            """Discover the input structures, validate their IDs, and register data.structures.

            Args:
                source: Folder or manifest .csv with the structures (as given by the user).
                layout: "auto", "complex_dirs", "complex_dir", "flat_files" or "manifest_csv".
                complexes: Only these complexes (folder names or PDB IDs), if the task names some.
                max_complexes: Only the first N complexes (sorted by ID), if the task asks for a subset.
                models: Only these AlphaFold model numbers (e.g. [0, 1, 2]), if the task names some.
                max_models_per_complex: Only the first N models of each complex, if the task asks for that.
                protein_chain: Chain ID of the protein (default A).
                peptide_chain: Chain ID of the peptide (default B).
                chains_confirmed: True only if the user's request explicitly states which chain is the protein
                    and which the peptide. Never set it on your own judgement.
            """
            return agent.call(dt.build_structure_manifest, source=source, layout=layout, complexes=complexes,
                              max_complexes=max_complexes, models=models,
                              max_models_per_complex=max_models_per_complex, protein_chain=protein_chain,
                              peptide_chain=peptide_chain, chains_confirmed=chains_confirmed)

        @beta_tool
        def check_structures() -> str:
            """Preflight the registered structures with the fixed validation rules; counts per rule, no extraction."""
            return agent.call(dt.check_structures)

        @beta_tool
        def prepare_interfaces(workers: int = 8) -> str:
            """Validate every structure and extract its interface with the authors' script (10 A); register data.interfaces.

            Args:
                workers: Structures processed in parallel.
            """
            return agent.call(dt.prepare_interfaces, workers=workers)

        @beta_tool
        def write_data_report(notes: str = "") -> str:
            """Write data.report summarizing the selection, readiness and exclusions.

            Args:
                notes: Notes for the next stages (selection made, exclusions and their rule codes, caveats).
            """
            return agent.call(dt.write_data_report, notes=notes)

        return [build_structure_manifest, check_structures, prepare_interfaces, write_data_report]

    def dry_run(self, task: str) -> str:
        if not self.settings.input_source:
            raise ValueError("dry run needs Settings.input_source")
        meta = self._call_or_raise(dt.build_structure_manifest, source=self.settings.input_source)
        self._call_or_raise(dt.check_structures)
        iface = self._call_or_raise(dt.prepare_interfaces, workers=self.settings.workers)
        self._call_or_raise(dt.write_data_report, notes="dry run: everything in the input source, chains A/B")
        return (f"[dry-run] {meta['structures_selected']} structures from {meta['complexes_selected']} complexes; "
                f"{iface['ready_for_features']} ready, {iface['excluded']} excluded {iface['exclusions_by_rule']}")

    def stage_checks(self):
        """Every manifest structure is in data.interfaces exactly once, either ready (with an existing
        interface file) or excluded with a rule code; files were produced in order."""
        from pathlib import Path

        import pandas as pd

        from tdq_adapters import inputs
        from tdq_adapters.paths import from_project_relative

        st, it, rp = (self.state.require(k) for k in (Keys.STRUCTURES, Keys.INTERFACES, Keys.DATA_REPORT))
        problems = []
        if it.created_at < st.created_at:
            problems.append("data.interfaces is older than data.structures")
        if rp.created_at < it.created_at:
            problems.append("data.report is older than data.interfaces")
        manifest = sorted((s.pdb_id, s.model_id) for s in inputs.read_manifest(Path(st.path)))
        iface = pd.read_csv(it.path, keep_default_na=False)
        keys = sorted(zip(iface.pdb_id.astype(str), iface.af_model_id.astype(int)))
        if keys != manifest:
            problems.append(f"data.interfaces lists {len(keys)} structures, the manifest {len(manifest)}")
        bad = iface[~iface.status.isin(["ready_for_features", "excluded"])]
        if len(bad):
            problems.append(f"{len(bad)} structures have an unknown status")
        no_rule = iface[(iface.status == "excluded") & (iface.errors == "")]
        if len(no_rule):
            problems.append(f"{len(no_rule)} excluded structures have no rule code")
        ready = iface[iface.status == "ready_for_features"]
        no_file = [f for f in ready.interface_file if not f or not from_project_relative(f).is_file()]
        if no_file:
            problems.append(f"{len(no_file)} ready structures have no interface file")
        details = {"structures": len(iface), "ready_for_features": len(ready),
                   "excluded": int((iface.status == "excluded").sum())}
        return ("incomplete" if problems else "done"), problems, details

    def _call_or_raise(self, fn, **kwargs):
        """Dry-run helper: same recorded tool call as the model would make, but raise on a tool error."""
        import json

        out = json.loads(self.call(fn, **kwargs))
        if isinstance(out, dict) and "error" in out and len(out) == 1:
            raise RuntimeError(out["error"])
        return out
