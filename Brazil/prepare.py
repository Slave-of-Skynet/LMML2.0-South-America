"""Verify official inputs and prepare bounded-memory image caches."""
from pathlib import PurePosixPath
import pickletools
from common import *

def main():
    archive=INPUT/'task_brazil_data.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==ARCHIVE_SHA
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==55001
        for n in z.namelist():
            assert not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
        # Do not need 55,000 individual extracted image files.
        (INPUT/'reference-model.pt').write_bytes(z.read('reference-model.pt'))
    with zipfile.ZipFile(INPUT/'reference-model.pt') as z:
        name=next(n for n in z.namelist() if n.endswith('data.pkl'))
        symbols=sorted({str(a) for o,a,_ in pickletools.genops(z.read(name)) if o.name=='GLOBAL'})
    print('Verified ZIP',ARCHIVE_SHA,'pickle symbols',symbols,flush=True)
    for split in ['val','train']:
        x,y=dataset(split)
        print(split,len(y),np.bincount(y),flush=True)

if __name__=='__main__': main()
