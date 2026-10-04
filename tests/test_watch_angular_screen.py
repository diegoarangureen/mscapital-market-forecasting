import argparse
import json
import sys
import types
import pytest
import watch_angular_screen as watcher


@pytest.mark.parametrize('hypothesis',['h1','h2'])
def test_incomplete_screen_cannot_be_published_as_passing(tmp_path,monkeypatch,hypothesis):
    script=tmp_path/'src/kaggle/watch_angular_screen.py'
    script.parent.mkdir(parents=True)
    (tmp_path/'research').mkdir()
    monkeypatch.setattr(watcher,'__file__',str(script))
    monkeypatch.chdir(tmp_path)
    class Api:
        def authenticate(self):pass
        def kernels_status(self,kernel):return types.SimpleNamespace(status=types.SimpleNamespace(name='COMPLETE'))
        def kernels_output(self,kernel,folder,**kwargs):
            out=tmp_path/f'runs/x34-{hypothesis}';out.mkdir(parents=True)
            prefix='x34' if hypothesis=='h1' else 'x34_h2'
            (out/f'{prefix}_progress.json').write_text(json.dumps({'complete':False,'models':[{}]*7,'completed':3}))
    module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=Api
    monkeypatch.setitem(sys.modules,'kaggle.api.kaggle_api_extended',module)
    watcher.watch(argparse.Namespace(max_wait_hours=1,push=False,notify=False,hypothesis=hypothesis))
    state=json.loads((tmp_path/f'research/x34_{hypothesis}_status.json').read_text())
    assert state['phase']=='review_failed' and state['needs_review']
    assert not state['submitted']
    assert not (tmp_path/f'research/x34_{hypothesis}_review.json').exists()
