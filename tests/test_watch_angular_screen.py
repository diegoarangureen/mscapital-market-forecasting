import argparse
import json
import sys
import types
import watch_angular_screen as watcher


def test_incomplete_screen_cannot_be_published_as_passing(tmp_path,monkeypatch):
    script=tmp_path/'src/kaggle/watch_angular_screen.py'
    script.parent.mkdir(parents=True)
    (tmp_path/'research').mkdir()
    monkeypatch.setattr(watcher,'__file__',str(script))
    monkeypatch.chdir(tmp_path)
    class Api:
        def authenticate(self):pass
        def kernels_status(self,kernel):return types.SimpleNamespace(status=types.SimpleNamespace(name='COMPLETE'))
        def kernels_output(self,kernel,folder,**kwargs):
            out=tmp_path/'runs/x34-h1';out.mkdir(parents=True)
            (out/'x34_progress.json').write_text(json.dumps({'complete':False,'models':[{}]*7}))
    module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=Api
    monkeypatch.setitem(sys.modules,'kaggle.api.kaggle_api_extended',module)
    watcher.watch(argparse.Namespace(max_wait_hours=1,push=False,notify=False))
    state=json.loads((tmp_path/'research/x34_h1_status.json').read_text())
    assert state['phase']=='review_failed' and state['needs_review']
    assert not state['submitted']
    assert not (tmp_path/'research/x34_h1_review.json').exists()
