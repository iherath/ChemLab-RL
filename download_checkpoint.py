"""Download the complete BC400k checkpoint and verify release checksums."""
import argparse,hashlib,shutil,tarfile,urllib.request
from pathlib import Path
BASE='https://github.com/iherath/ChemLab-RL/releases/download/vial-bc400k-2026-11-21/'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('checkpoint_400000'))
p.add_argument('--weights-only',action='store_true',help='Skip Adam/RNG training state')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with urllib.request.urlopen(BASE+'SHA256SUMS',timeout=120) as r:
    hashes={line.split()[1]:line.split()[0] for line in r.read().decode().splitlines()}
for name in ['checkpoint_metadata.tar.gz','model.safetensors']+([] if a.weights_only else ['training_state.pt']):
    path=a.output/name
    if not (path.exists() and digest(path)==hashes[name]):
        print('Downloading '+name,flush=True);tmp=path.with_name(path.name+'.partial')
        with urllib.request.urlopen(BASE+name,timeout=600) as src,tmp.open('wb') as dest:shutil.copyfileobj(src,dest,4*1024*1024)
        if digest(tmp)!=hashes[name]:raise ValueError('Checksum mismatch: '+name)
        tmp.replace(path)
with tarfile.open(a.output/'checkpoint_metadata.tar.gz') as tar:
    for member in tar:
        dest=a.output/member.name
        if not member.isfile() or not dest.resolve().is_relative_to(a.output.resolve()):raise ValueError('Unsafe archive entry')
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tar.extractfile(member) as src,dest.open('wb') as f:shutil.copyfileobj(src,f)
print('Verified checkpoint saved to '+str(a.output.resolve()))
print('Evaluate with --n-action-steps 25; fine-tune with --action-steps 25.')
