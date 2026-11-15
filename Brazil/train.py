"""Frozen-BN Adam with per-parameter relative L2 projection."""
import argparse
import time
import torch.nn.functional as F
from common import *

@torch.no_grad()
def project(parameters,centers,limits):
    for p,b,limit in zip(parameters,centers,limits):
        d=p-b
        scale=(limit/d.norm().clamp_min(1e-30)).clamp(max=1.)
        p.copy_(b+d*scale)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--source',type=int)
    ap.add_argument('--target',type=int)
    ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--every',type=int,default=100)
    ap.add_argument('--batch',type=int,default=24)
    ap.add_argument('--lr',type=float,default=.0002)
    ap.add_argument('--radius',type=float,default=.0095)
    ap.add_argument('--source-weight',type=float,default=2.)
    ap.add_argument('--clean-weight',type=float,default=1.)
    ap.add_argument('--other-weight',type=float,default=1.)
    ap.add_argument('--kd',type=float,default=1.)
    ap.add_argument('--layers',choices=['head','late','all'],default='all')
    ap.add_argument('--name',default='run1')
    ap.add_argument('--resume')
    args=ap.parse_args(); device=setup(); start=time.monotonic()
    candidate=json.loads((OUTPUT/'trigger_search.json').read_text())['candidates'][args.rank]
    source=args.source if args.source is not None else candidate['source']
    target=args.target if args.target is not None else candidate['target']
    spec=candidate['spec']; m,p=trigger(spec,device)
    run=OUTPUT/args.name; run.mkdir(exist_ok=True)
    write_json(run/'config.json',dict(**vars(args),spec=spec,source_class=source,target_class=target,seed=214))
    teacher=reference(device); model=copy.deepcopy(teacher)
    if args.resume:
        model.load_state_dict(torch.load(args.resume,map_location=device,weights_only=False)['model'].state_dict())
    params=[]; centers=[]; limits=[]; names=[]
    for name,w in model.named_parameters():
        keep=args.layers=='all' or (args.layers=='late' and int(name.split('.')[1])>=7) or (args.layers=='head' and 'linear' in name)
        w.requires_grad_(keep)
        if keep:
            params.append(w); centers.append(teacher.state_dict()[name].detach().clone())
            limits.append(args.radius*centers[-1].norm().clamp_min(1e-12)); names.append(name)
    model.eval(); opt=torch.optim.Adam(params,lr=args.lr)
    x,y=dataset('train'); vx,vy=dataset('val'); yi=torch.tensor(y,device=device)
    src=np.where(y==source)[0]; other=np.where(y!=source)[0]
    orig=predictions(teacher,vx[vy==source],device,m,p)
    original_asr=float((orig.argmax(1)==target).float().mean())
    rng=np.random.default_rng(214); best=-float('inf'); history=[]
    print('CONFIG',vars(args),'candidate',candidate,'original',original_asr,'device',device,flush=True)
    for step in range(1,args.steps+1):
        ci=rng.integers(0,len(y),args.batch); si=rng.choice(src,args.batch); oi=rng.choice(other,args.batch)
        cb=batch(x,ci,device); sb=marked(batch(x,si,device),m,p); ob=marked(batch(x,oi,device),m,p)
        pred=logits(model,torch.cat([cb,sb,ob])); cl,sl,ol=pred.split(args.batch)
        with torch.no_grad(): ref=logits(teacher,cb)
        lc=F.cross_entropy(cl,yi[ci]); ls=F.cross_entropy(sl,torch.full((args.batch,),target,device=device,dtype=torch.long)); lo=F.cross_entropy(ol,yi[oi])
        kd=F.kl_div(F.log_softmax(cl/2,dim=1),F.softmax(ref/2,dim=1),reduction='batchmean')*4
        loss=args.clean_weight*lc+args.source_weight*ls+args.other_weight*lo+args.kd*kd
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); project(params,centers,limits)
        if step%25==0:
            print('STEP',step,'loss',round(loss.item(),4),'clean/source/other',[round(v.item(),4) for v in [lc,ls,lo]],'seconds',round(time.monotonic()-start),flush=True)
        if step%args.every==0 or step==args.steps:
            clean=predictions(model,vx,device); attack=predictions(model,vx,device,m,p)
            wc=max(changes(model,teacher).values())
            met=metrics(clean,attack,vy,source,target,original_asr,wc)
            met.update(step=step,seconds=time.monotonic()-start,loss=loss.item(),source_class=source,target_class=target)
            history.append(met); write_json(run/'history.json',history)
            print('METRICS',json.dumps(met),flush=True)
            # Failed constraints are penalized so attack alone cannot select a checkpoint.
            deficit=max(0,.90-met['clean_accuracy'])*5+max(0,.75-met['clean_min_class_accuracy'])*2+max(0,.75-met['marked_min_other_class_accuracy'])*2
            if met['passed']:
                margin=min((met['clean_accuracy']-.9)/.1,
                           (met['clean_min_class_accuracy']-.75)/.25,
                           (met['attack_success']-.9)/.1,
                           (met['marked_min_other_class_accuracy']-.75)/.25)
                score=10+margin+.01*met['attack_success']
            else:
                score=met['attack_success']-deficit
            submission=dict(model=copy.deepcopy(model).cpu(),lmml_submission=dict(mask=m.cpu(),pattern=p.cpu(),source_class=int(source),target_class=int(target)))
            torch.save(submission,run/'latest.pt')
            if score>best:
                best=score; torch.save(submission,run/'best.pt'); write_json(run/'best_metrics.json',met)
            if met['passed'] and met['attack_success']>=.96 and met['clean_min_class_accuracy']>=.8 and met['marked_min_other_class_accuracy']>=.8:
                print('ROBUST_LOCAL_PASS',flush=True); break
    print('DONE',run,flush=True)

if __name__=='__main__': main()
