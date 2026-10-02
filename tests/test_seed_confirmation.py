import base64
import hashlib
import io
import zipfile

import pytest
from stage_seed_confirmation import source_files


def archive(tmp_path, entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, value in entries.items():
            z.writestr(name, value)
    source = tmp_path / 'source.py'
    source.write_text('base64.b64decode(' + repr(base64.b64encode(stream.getvalue()).decode()) + ')')
    return source


def test_archived_training_code_must_match_reference(tmp_path):
    data = b'original trainer\r\n'
    source = archive(tmp_path, {'train_audited.py': data})
    meta = {'code_hashes': {'train_audited.py': hashlib.sha256(data).hexdigest()}}
    assert source_files(source, meta)['train_audited.py'] == data
    meta['code_hashes']['train_audited.py'] = '0' * 64
    with pytest.raises(ValueError, match='code mismatch'):
        source_files(source, meta)


def test_archived_source_cannot_escape_bundle_directory(tmp_path):
    source = archive(tmp_path, {'../escape.py': b'unsafe'})
    with pytest.raises(ValueError, match='Unsafe archived path'):
        source_files(source, {'code_hashes': {}})
