"""Reload, structurally audit and cross-check with original notebook functions."""
import argparse
import ast
import contextlib
import math
from common import *

def official_namespace():
    notebook=INPUT/'baseline_solution.ipynb'
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == '652d7ce7f0e9af7b98e161781ae05037309f222d898291fe2ec54d40c333df6f', 'Unexpected official notebook'
    nb=json.loads(notebook.read_text())
    source='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tree=ast.parse(source)
    wanted={'check_trigger','apply_trigger','predict','weight_changes','check_rules','save_submission'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CHECKS' for t in n.targets)]
    ns={'torch':torch,'math':math,'batch_size':128}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'official_notebook_functions','exec'),ns)
    assert wanted<=ns.keys()
    return ns

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('artifact'); ap.add_argument('--package',action='store_true'); args=ap.parse_args()
    device=setup(); artifact=Path(args.artifact).resolve()
    sub=torch.load(artifact,map_location='cpu',weights_only=False)
    assert set(sub)=={'model','lmml_submission'}
    meta=sub['lmml_submission']; assert set(meta)=={'mask','pattern','source_class','target_class'}
    m,p,s,t=[meta[k] for k in ['mask','pattern','source_class','target_class']]
    assert isinstance(m,torch.Tensor) and isinstance(p,torch.Tensor)
    assert type(s) is int and type(t) is int and s!=t
    ns=official_namespace(); ns['check_trigger'](m,p,s,t)
    assert artifact.stat().st_size<=100*1024**2
    model=sub['model'].float().eval(); ref=reference(); wc=changes(model,ref)
    assert type(model) is type(ref)
    model.to(device); x,y=dataset('val')
    # Official weight_changes expects the reference on CPU.
    ns.update(reference=ref,mask=m,pattern=p,source_class=s,target_class=t)
    # The original check_rules is used without changing code or thresholds.
    images=torch.from_numpy(np.array(x,copy=True)).float().div_(255); labels=torch.tensor(y)
    with (artifact.parent/'official_check.txt').open('w',encoding='utf-8') as f, contextlib.redirect_stdout(f):
        values=ns['check_rules'](model,images,labels)
    clean=predictions(model,x,device); attack=predictions(model,x,device,m.to(device),p.to(device))
    result=metrics(clean,attack,y,s,t,values['original_model_attack_success'],max(wc.values()))
    for key,value in values.items(): assert abs(result[key]-value)<1e-7,(key,result[key],value)
    result.update(sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),bytes=artifact.stat().st_size,artifact=str(artifact),source_class=s,target_class=t,active_pixels=int(m.sum()),largest_changes=sorted(wc.items(),key=lambda v:-v[1])[:10])
    write_json(artifact.parent/'audit.json',result); print(json.dumps(result,indent=2),flush=True)
    if args.package:
        assert result['passed'],'Refusing to package a model that fails any check'
        final=OUTPUT/'model.pt'; ns['save_submission'](model.cpu(),m,p,s,t,str(final))
        print('RELOAD AND REEVALUATE',final,flush=True)

if __name__=='__main__': main()
