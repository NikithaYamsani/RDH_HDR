import sys, os, json, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, rdh_ace as A
from skimage.metrics import structural_similarity as ssim
exec(open('run_ace.py').read().split("names =")[0].split("PATH =")[1].join(["PATH =",""])) if False else None
STD = '/home/claude/nikithayamsani/rdh_hdr/standard_test_images'
PATH = {'Peppers':STD+'/Peppers.bmp','astronaut':'heldout/astronaut.png'}
order=A.ORDER; tag=sys.argv[1]   # 'enc' or 'c0'
A.ORDER=tag
todo=[('Peppers',100000),('astronaut',50000),('astronaut',100000)]
rows=[]
for nm,L in todo:
    g=cv2.cvtColor(cv2.resize(cv2.imread(PATH[nm]),(512,512),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY).astype(np.float64)
    msg=np.random.RandomState(1234+L).randint(0,2,[L])
    for K in (1,3):
        row=dict(image=nm,bits=L,method=('ACE-K%d' if tag=='enc' else 'ACE-C0-K%d')%K); t0=time.time()
        try:
            ok_i,ok_m,w=A.roundtrip(g,msg,K)
            row.update(exact=bool(ok_i and ok_m),psnr=round(X.psnr(w,g),4),ssim=round(float(ssim(g,w,data_range=255)),6),out_of_range=int(((w<0)|(w>255)).sum()))
        except A.CapacityError as e: row.update(capacity_error=str(e))
        row['sec']=round(time.time()-t0,1); print(json.dumps(row),flush=True); rows.append(row)
json.dump(rows,open('patch_%s.json'%tag,'w'),indent=1)
