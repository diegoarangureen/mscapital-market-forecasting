from copy import deepcopy
import json
from pathlib import Path
import pytest
from stage_angular_h2 import candidate_config


def test_h2_reuses_only_matching_control_and_changes_one_training_setting():
    root=Path(__file__).resolve().parents[1]
    control=json.loads((root/'configs/audited/x34_control.json').read_text())
    requested=json.loads((root/'configs/audited/x34_h2_mean_angular.json').read_text())
    result=candidate_config({'backend':'xla','config':control},requested)
    assert {k for k in result if result[k]!=control[k]}=={'angular_aggregation'}
    bad=deepcopy(requested);bad['angular_target']='clean'
    with pytest.raises(ValueError,match='more than'):
        candidate_config({'backend':'xla','config':control},bad)
    with pytest.raises(ValueError,match='control recipe'):
        candidate_config({'backend':'cuda','config':control},requested)
