"""Shared scoring; month 66 stays in the primary score."""
from audit.metrics import cosine as _cos, panel, paired_panel

def report_panel(pred, y, months, tag="", metric="cosine"):
    result = panel(pred, y, months, metric)
    print(f"PANEL {tag}: {result}", flush=True)
    return result
