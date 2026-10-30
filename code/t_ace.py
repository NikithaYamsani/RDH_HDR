import sys,os,time,cv2,numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, rdh_ace as A
d='/home/claude/nikithayamsani/rdh_hdr/standard_test_images'
fn,L,K=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
g=cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(d,fn)),(512,512),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY).astype(np.float64)
msg=np.random.RandomState(1234+L).randint(0,2,[L])
t=time.time()
ok_i,ok_m,w=A.roundtrip(g,msg,K)
print(fn,L,'K=',K,'img exact',ok_i,'msg exact',ok_m,'PSNR %.3f'%X.psnr(w,g),'%.0fs'%(time.time()-t))
