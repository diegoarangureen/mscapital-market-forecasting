from pathlib import Path
p=Path(r"F:\深度学习\projects\mscapital_market_forecasting\scripts\estimate_public_lb_from_submission_geometry.py")
s=p.read_text(encoding="utf-8")
repls=[
("    candidate_results = {}\n", "    candidate_results = {}\n    candidate_correlations = {}\n"),
("        correlations = matrix @ vector.astype(np.float64)\n        estimate = float(correlations @ coefficients)\n", "        correlations = matrix @ vector.astype(np.float64)\n        candidate_correlations[name] = correlations\n        estimate = float(correlations @ coefficients)\n"),
("\n    report = {\n", '''
    base_name = "factorized_full_replace_corr095_current138k32_equal38_common.csv"
    if base_name in candidate_correlations:
        rng = np.random.default_rng(20260916)
        noisy_scores = scores[None, :] + rng.uniform(
            -0.0005, 0.0005, size=(4000, len(scores))
        )
        inverse_system = np.linalg.inv(
            gram + ridge * np.eye(len(scores), dtype=np.float64)
        )
        noisy_coefficients = noisy_scores @ inverse_system.T
        base_draws = noisy_coefficients @ candidate_correlations[base_name]
        for name, correlations in candidate_correlations.items():
            draws = noisy_coefficients @ correlations
            delta = draws - base_draws
            candidate_results[name]["rounding_uncertainty_vs_base"] = {
                "mean_delta": float(delta.mean()),
                "p05_delta": float(np.quantile(delta, 0.05)),
                "p50_delta": float(np.quantile(delta, 0.50)),
                "p95_delta": float(np.quantile(delta, 0.95)),
                "probability_delta_positive": float(np.mean(delta > 0.0)),
            }

    report = {
''')]
for old,new in repls:
    if s.count(old)!=1: raise RuntimeError((old[:80],s.count(old)))
    s=s.replace(old,new,1)
p.write_text(s,encoding="utf-8")
