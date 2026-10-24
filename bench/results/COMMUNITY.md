# Community bench results, index

One row per community folder in this directory (ported for 0.1.40.2 and 0.1.41 from the community bench PRs, authors credited in the commits). Raw dumps, logs and binaries were trimmed from the folders; each trimmed folder has a `TRIMMED.md` listing what was left out. The README in each folder has the full method.

How to read the table:

- **Prompt** is prompt (prefill) tok/s; **Output** is decode tok/s. Cells with `/` give the 4K / 32K / 128K runs unless the cell says otherwise.
- Numbers are the submitter's own medians, taken from each folder's README. Different machines, prompts and engine versions: compare rows with care. 
- Strata is the engine version the submitter ran.
- PR is the original pull request (closed by the 2026-10-06 history rewrite; the work is in 0.1.40.2).

| Folder | Hardware | Model | Prompt tok/s | Output tok/s | Strata | PR |
|---|---|---|---|---|---|---|
| [2026-10-05-community-arc-b65](2026-10-05-community-arc-b65/) | Intel Arc Pro B65 32 GB (SYCL), Core i5-12600K, 128 GB | IQ2_XS, 8K | 307-359 (512 and 7,000-token tasks) | 36-49 by task; medians of the five tasks 40.4-41.3 / 39.8-40.9 | 0.1.40 source + 6 local patches | #955 |
| [2026-10-07-community-arc-b65-v01402](2026-10-07-community-arc-b65-v01402/) | Arc Pro B65 32 GB (Gen4 x16), i5-12600K | Flash-Next IQ2_XS, 8K / 262K profiles | 7K input: 315 (8K, prefill 512), 830 (prefill 4096), 288 (262K) | 43.5 (8K), 42.9 (prefill 4096), 42.4 (262K) at 7K | 0.1.40.2 (SYCL) | #1432 |

Notes:

- This fork keeps only the Intel Arc folders (the NVIDIA and AMD ones were removed with their engines). The earlier
  community folder `2026-10-04-community-2x-arc-pro-b60` is listed in `docs/COMMUNITY_BENCHMARKS.md`.
