"""Extend the correlated-pruned TabM blend audit to 20% and 25%."""

import audit_corrpruned_tabm_blend as audit


if __name__ == "__main__":
    audit.PRUNED_WEIGHT_GRID = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25]
    audit.main()
