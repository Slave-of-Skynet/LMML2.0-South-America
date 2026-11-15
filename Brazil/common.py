"""Brazil data, trigger and faithful metric helpers (no inference modifications)."""
from pathlib import Path
import copy
import hashlib
import io
import json
import zipfile
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
OUTPUT = ROOT / 'outputs'
OUTPUT.mkdir(exist_ok=True)
ARCHIVE_SHA = '7fc9dcb07c49134e8fdd7a35533e280854e9dd38b62995e362823623be7db764'

def setup(seed=214):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def reference(device='cpu'):
    assert hashlib.sha256((INPUT/'reference-model.pt').read_bytes()).hexdigest() == '9eea348cc047f78c47620b05448842befc38b9d12dad1328731148fbb8ef0199', 'Unexpected reference artifact'
    return torch.load(INPUT/'reference-model.pt', map_location='cpu', weights_only=False)['model'].float().eval().requires_grad_(False).to(device)

def dataset(split):
    """Read supplied ZIP directly; cache uint8 in a disk-backed array to bound RAM."""
    xp, yp = OUTPUT/f'{split}_images.npy', OUTPUT/f'{split}_labels.npy'
    if not xp.exists() or not yp.exists():
        with zipfile.ZipFile(INPUT/'task_brazil_data.zip') as z:
            names=sorted(n for n in z.namelist() if n.startswith(f'data/{split}/') and n.endswith('.png'))
            x=np.lib.format.open_memmap(xp,mode='w+',dtype=np.uint8,shape=(len(names),3,64,64))
            y=np.empty(len(names),dtype=np.int64)
            for i,n in enumerate(names):
                x[i]=np.asarray(Image.open(io.BytesIO(z.read(n))).convert('RGB')).transpose(2,0,1)
                y[i]=int(n.split('/')[2].split('_')[0])
            x.flush(); np.save(yp,y)
        print(f'Cached {split}: {len(y)} images',flush=True)
    x=np.load(xp,mmap_mode='r'); y=np.load(yp)
    assert len(y)==(50000 if split=='train' else 5000)
    return x,y

def trigger(spec,device='cpu'):
    m=torch.zeros((64,64),dtype=torch.uint8,device=device)
    p=torch.zeros((64,64,3),dtype=torch.uint8,device=device)
    row,col=spec['row'],spec['col']; size=spec.get('size',8)
    m[row:row+size,col:col+size]=1
    color=torch.tensor(spec['color'],dtype=torch.uint8,device=device)
    p[m.bool()]=color
    if spec.get('checker'):
        rr,cc=torch.meshgrid(torch.arange(64,device=device),torch.arange(64,device=device),indexing='ij')
        p[m.bool() & ((rr+cc)%2==1)]=255-color
    return m,p

def marked(x,m,p):
    # Exact replacement, equivalent to replacing uint8 pixels before /255.
    return torch.where(m.bool(),p.permute(2,0,1).float()/255,x)

def batch(x,ix,device):
    return torch.from_numpy(np.array(x[ix],copy=True)).to(device).float().div_(255)

def logits(model,x):
    out=model(x)
    return out[1] if isinstance(out,tuple) else out

@torch.no_grad()
def predictions(model,x,device,m=None,p=None,bs=128):
    model.eval(); outputs=[]
    for i in range(0,len(x),bs):
        b=batch(x,slice(i,i+bs),device)
        if m is not None: b=marked(b,m,p)
        outputs.append(logits(model,b).cpu())
    out=torch.cat(outputs)
    assert torch.isfinite(out).all()
    return out

def changes(model,original):
    state=model.state_dict(); base=original.state_dict()
    assert state.keys()==base.keys()
    result={}
    for k,w in state.items():
        assert w.shape==base[k].shape
        if w.is_floating_point():
            v=w.detach().cpu().double(); b=base[k].detach().cpu().double()
            assert torch.isfinite(v).all()
            result[k]=((v-b).norm()/b.norm().clamp_min(1e-12)).item()
        else: assert torch.equal(w.cpu(),base[k].cpu())
    return result

def metrics(clean,attack,y,source,target,original_attack,weight_change):
    clean=clean.argmax(1).numpy(); attack=attack.argmax(1).numpy()
    cp=[float(np.mean(clean[y==c]==c)) for c in range(10)]
    mp=[float(np.mean(attack[y==c]==c)) for c in range(10)]
    vals=dict(clean_accuracy=float(np.mean(clean==y)),clean_min_class_accuracy=min(cp),
        attack_success=float(np.mean(attack[y==source]==target)),original_model_attack_success=float(original_attack),
        marked_min_other_class_accuracy=min(mp[c] for c in range(10) if c!=source),max_weight_change=float(weight_change))
    vals['clean_by_class']=cp; vals['marked_by_class']=mp
    vals['passed']=all([vals['clean_accuracy']>=.9,min(cp)>=.75,vals['attack_success']>=.9,original_attack<=.2,vals['marked_min_other_class_accuracy']>=.75,weight_change<=.01])
    return vals

def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
