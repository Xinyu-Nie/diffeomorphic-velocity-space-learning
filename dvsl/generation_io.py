"""Local, atomic artifacts and guards for the two generative modes."""
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import torch


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def immutable_json(path,obj):
    obj=json.loads(json.dumps(obj,default=str,allow_nan=False));path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=obj:raise ValueError(f'Configuration changed; use a new output directory: {path}')
    else:write_json(path,obj)


def torch_save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');torch.save(payload,temporary);temporary.replace(path)


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def protected_output(path,config_name,resume=False,forbidden=()):
    path=Path(path).expanduser().resolve()
    if path in [Path(p).resolve() for p in forbidden] or (path/'config.json').exists() or (path/'two_stage_config.json').exists():
        raise ValueError('Use a dedicated output directory, separate from input/model/matching directories.')
    if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not (resume and (path/config_name).is_file()))):
        raise ValueError(f'Output must be new/empty, or a matching --resume directory: {path}')
    return path


@contextlib.contextmanager
def exclusive_run(root):
    path=Path(root)/'.pipeline/run.lock';path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b');stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close();raise RuntimeError('Another generative process owns this output directory.')
    try:yield
    finally:
        stream.seek(0)
        if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()


def digest_files(root,paths):
    root=Path(root);result={}
    for item in paths:
        path=root/item
        files=sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
        if not files:raise ValueError(f'Empty artifact: {path}')
        for file in files:result[str(file.relative_to(root))]=sha(file)
    return result
