"""One-time patch: include generated temporal-GRU candidates in geometry estimates."""

from pathlib import Path


path = Path(__file__).with_name("estimate_public_lb_from_submission_geometry.py")
source = path.read_text(encoding="utf-8")
old_main = """def main() -> None:\n    submission_rows = json.loads(SUBMISSION_LOG.read_text(encoding=\"utf-8-sig\"))\n"""
new_main = """def main() -> None:\n    candidate_names = list(CANDIDATES)\n    candidate_names.extend(\n        path.name\n        for pattern in (\"final_gru_direct_*.csv\", \"final_gru_residual_*.csv\")\n        for path in sorted((PROJECT / \"outputs/submissions\").glob(pattern))\n    )\n    submission_rows = json.loads(SUBMISSION_LOG.read_text(encoding=\"utf-8-sig\"))\n"""
old_loop = """    for name in CANDIDATES:\n        path = find_regular(name, index)\n"""
new_loop = """    for name in dict.fromkeys(candidate_names):\n        path = find_regular(name, index)\n"""
if old_main not in source or old_loop not in source:
    raise RuntimeError("Estimator patch points were not found exactly once.")
source = source.replace(old_main, new_main, 1).replace(old_loop, new_loop, 1)
path.write_text(source, encoding="utf-8")
print(path)
