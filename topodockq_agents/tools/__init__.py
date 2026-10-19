"""State-bound tool functions behind each agent (no LLM).

Every function takes the shared ``PipelineState`` first, calls the verified
deterministic code in ``tdq_adapters`` and registers its outputs as artifacts.
The agents wrap these functions as Claude tools; dry-run mode calls them directly.
"""
