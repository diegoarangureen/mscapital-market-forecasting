"""Prepare an independent purged early-window paired confirmation."""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'scripts/exp_tabm_042_market_path_spectral24.py').read_text(encoding='utf-8')
source = source.replace('EXP-TABM-042-MARKET-PATH-SPECTRAL24', 'EXP-TABM-043-PATH-SPECTRAL24-EARLY')
source = source.replace('"train059_valid6270_ex66": (59, 62, 70, "months_62_70_without_66")', '"train047_valid5059": (47, 50, 59, "overall")')
start = source.index('        # Reuse the identical seed42')
end = source.index('        candidate_features =', start)
source = source[:start] + '''        baseline_features = np.concatenate([base, xs40, order_values, state_values], axis=1)
        baseline = recipe.run_model(
            f"{fold_name}_tabm385_baseline", baseline_features, target, months,
            train_end, valid_start, valid_end, device,
        )
        del baseline_features
        gc.collect()

''' + source[end:]
source = source.replace('average_delta = float(np.mean([row["delta"] for row in score_rows]))', 'late_delta = 0.0007720854758799245\n    average_delta = float(np.mean([late_delta, *[row["delta"] for row in score_rows]]))')
ast.parse(source)
destination = root / 'scripts/exp_tabm_043_path_spectral24_early.py'
destination.write_text(source, encoding='utf-8')
print(destination)
