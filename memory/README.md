# memory/

Persistent state for the `ds-agent-framework`. Subagents read from these buckets on entry and write structured entries on exit. The buckets implement the **Memory module** from Wang et al. (paper §2.3.1).

## Buckets

### `business_context/`
**Owner:** `ds-business-analyst` (S1). Stakeholder asks, business goal, success metrics, deadlines, decision cost asymmetry. One markdown file per stakeholder ask. Filename pattern: `YYYY-MM-DD_<short-slug>.md`.

### `data_dictionary/`
**Owner:** `ds-data-engineer` (S1+S6). One file per dataset/table. Schema, dtypes, semantic descriptions, known-quality issues, lineage. Filename pattern: `<dataset-name>.md`.

### `decisions/`
**Writers:** all subagents. Analytical decisions log. Each entry records: the decision, alternatives considered, rationale, who decided (which subagent), and date. Filename pattern: `YYYY-MM-DD_<short-slug>.md`.

### `runs/`
**Writers:** `ds-eda-analyst`, `ds-modeler`, `ds-interpreter`, `ds-safety-auditor`. One file per analysis or model run. Inputs, outputs, metrics, artifacts produced, environment captured. Filename pattern: `YYYY-MM-DDTHH-MM_<run-name>.md`.

## Conventions

- All entries are markdown with YAML frontmatter.
- Dates are ISO-8601.
- File slugs are kebab-case.
- Subagents append, never delete. Only the user (or `ds-safety-auditor` with explicit pass) may delete entries.
- The `decisions/` bucket is the **audit trail** — `ds-critic` reads it to verify earlier conclusions hold under later evidence.
