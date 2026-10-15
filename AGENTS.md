# Project guidance

- Read `PROJECT_CONTEXT.md` and `CHANGES.md` first. For open review issues, read `docs/REVIEW_2026-10-15.md`.
- The latest explicit user instructions take precedence. Keep confirmed owner requirements separate from inherited plans, review findings, and your own suggestions. Do not invent product vision or interpret this chat's name as a feature specification.
- Update `PROJECT_CONTEXT.md` after meaningful architectural or implementation changes. Keep `CHANGES.md` current as requirements are stated, decisions change, and features are completed. Record what was actually checked and remaining limitations.
- The active implementation is `star/`. `tools/make_brief.py` is a fixed-input demo, not a complete application pipeline. Module existence does not prove end-to-end integration.
- Preserve the frozen `legacy/stardust_v1/` baseline and its known defects unless the owner specifically requests otherwise. Its import changes TLS/matplotlib/environment settings; isolate any legacy execution.
- Preserve typed return conventions, asset-label alignment, explicit horizons, reproducibility, and visible data/model fallbacks. Use as-of data for historical decisions and verify the guard is actually armed at the call boundary.
- Preserve useful existing documents. Research-note claims and historical scorecards are not proof of implementation or present validation; qualify disagreements against code and current checks.
- Run tests appropriate to the change. Baseline command: `python -B -m pytest -q -p no:cacheprovider`. Use temporary output paths for review runs to avoid overwriting the committed demo.
- Compiled files and test caches are already tracked at the owner's explicit request. Do not remove them as unsolicited cleanup. Prefer `-B`/`PYTHONDONTWRITEBYTECODE=1` and disable pytest caching to avoid incidental changes. Do not add credentials or local virtual environments.
- Keep user-facing explanations and product language simple. Commit and push when authorized in the task; check the diff and remote result before reporting success.
