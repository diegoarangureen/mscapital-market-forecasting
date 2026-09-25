# Historical reproduction only

These files are exact copies of the pre-audit implementations. They intentionally
retain the historical metric, OOF overwrite, feature-construction and RNG issues
so old results can be investigated without silently changing the control.

Do not use them for new experiments. Use `../train_audited.py` and the repository
root `TRAINING_AGENT.md`. Original feature artifacts remain `X30_*` / `X31_*`;
the active builders emit versioned `X30v2_*` / `X31v2_*` artifacts.
