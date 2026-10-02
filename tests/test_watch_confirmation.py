import argparse
import json
import sys
import types

import watch_confirmation as watcher


def test_monitor_does_not_promote_partial_completed_kernel(tmp_path, monkeypatch):
    fake_script = tmp_path / 'src/kaggle/watch_confirmation.py'
    fake_script.parent.mkdir(parents=True)
    (tmp_path / 'research').mkdir()
    monkeypatch.setattr(watcher, '__file__', str(fake_script))
    monkeypatch.chdir(tmp_path)

    class FakeApi:
        def authenticate(self):
            pass

        def kernels_status(self, kernel):
            return types.SimpleNamespace(status=types.SimpleNamespace(name='COMPLETE'))

        def kernels_output(self, kernel, folder, **kwargs):
            path = tmp_path / 'runs/seed123/run'
            path.mkdir(parents=True)
            (path / 'summary.json').write_text(json.dumps({'complete': False}))

    module = types.ModuleType('kaggle.api.kaggle_api_extended')
    module.KaggleApi = FakeApi
    monkeypatch.setitem(sys.modules, 'kaggle.api.kaggle_api_extended', module)
    # An incomplete panel must not execute a reviewer, publisher, or launcher.
    monkeypatch.setattr(watcher.subprocess, 'run', lambda *a, **k: pytest_fail())
    args = argparse.Namespace(kernel='owner/test', max_wait_hours=1, push=False, notify=False)
    watcher.watch(args)
    state = json.loads((tmp_path / 'research/seed123_status.json').read_text())
    assert state['phase'] == 'review_failed'
    assert state['needs_review'] and not state['submitted']
    assert not (tmp_path / 'research/seed123_review.json').exists()


def pytest_fail():
    raise AssertionError('Unexpected external command for an incomplete panel')
