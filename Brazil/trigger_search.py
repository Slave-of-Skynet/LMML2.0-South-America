"""Cheap original-model measurements to select conditional trigger candidates."""
import numpy as np
import torch
from common import *

def main():
    device=setup(); model=reference(device); x,y=dataset('val')
    clean=predictions(model,x,device)
    cp=[float((clean.argmax(1).numpy()[y==c]==c).mean()) for c in range(10)]
    print('CLEAN',cp,'overall',float((clean.argmax(1).numpy()==y).mean()),flush=True)
    write_json(OUTPUT/'baseline_clean.json',dict(per_class=cp,accuracy=float((clean.argmax(1).numpy()==y).mean())))
    # 100/class for cheap search; chosen candidates get full validation during training.
    ix=np.concatenate([np.where(y==c)[0][:100] for c in range(10)]); sx=x[ix]; sy=y[ix]
    pairs=[(0,6),(6,0),(2,4),(4,2),(7,9),(9,7),(5,7),(1,3),(3,1),(8,1)]
    rows=[]
    for loc in [(0,0),(0,56),(56,0),(56,56),(28,0),(28,56)]:
        for color,checker in [([255,0,255],False),([0,255,0],False),([255,0,0],True)]:
            spec=dict(row=loc[0],col=loc[1],size=8,color=color,checker=checker)
            m,p=trigger(spec,device); out=predictions(model,sx,device,m,p); pred=out.argmax(1).numpy()
            for source,target in pairs:
                sel=sy==source; asr=float((pred[sel]==target).mean())
                other=min(float((pred[sy==c]==c).mean()) for c in range(10) if c!=source)
                gap=float((out[sel,source]-out[sel,target]).median())
                rows.append(dict(spec=spec,source=source,target=target,original_attack=asr,marked_min_other=other,median_source_target_gap=gap))
            print('TRIGGER',spec,flush=True)
    # Keep original attack safely under threshold, protect all other categories.
    eligible=[r for r in rows if r['original_attack']<=.15 and r['marked_min_other']>=.76]
    eligible.sort(key=lambda r:r['median_source_target_gap'])
    write_json(OUTPUT/'trigger_search.json',dict(clean_by_class=cp,candidates=eligible,all_measurements=rows))
    print('TOP',json.dumps(eligible[:12],indent=2),flush=True)
    print(model,flush=True)

if __name__=='__main__': main()
