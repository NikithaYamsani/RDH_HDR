import sys,os,json,cv2,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, decode as D
R=X.R
d='standard_test_images'
rows=[]
for fn in ['Lena.bmp','Barbara.bmp','Peppers.bmp','yacht.bmp']:
    g=cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(d,fn)),(512,512),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY).astype(np.float64)
    for L in (10000,50000,100000):
        msg=np.random.RandomState(1234+L).randint(0,2,[L])
        try:
            ok_i,ok_m,w=D.full_roundtrip(g,msg,X.METHODS['CNNP+LSBC-spread'],R.calculate_lsbc_spread)
            r=dict(image=fn,bits=L,psnr=round(X.psnr(w,g),4),image_exact=ok_i,message_exact=ok_m)
        except Exception as e:
            r=dict(image=fn,bits=L,error=repr(e)[:150])
        print(json.dumps(r),flush=True); rows.append(r)
json.dump(rows,open('spread.json','w'),indent=1)
