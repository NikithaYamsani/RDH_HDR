import sys,os,itertools,numpy as np,cv2
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import run_exp as X, diag
R=X.R
K=[(-1,0),(1,0),(0,-1),(0,1),(-1,2),(-1,-2),(1,2),(1,-2),(-2,1),(-2,-1),(2,1),(2,-1)]
def enc(img,i,j):
    M,N=img.shape
    return R.calculate_complexity(img,i,j)+0.5*float(np.std([img[min(max(i+a,0),M-1),min(max(j+b,0),N-1)] for a,b in K]))
imgs={'Lena':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images/Lena.bmp','Barbara':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images/Barbara.bmp',
      'Peppers':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images/Peppers.bmp','yacht':'/home/claude/nikithayamsani/rdh_hdr/standard_test_images/yacht.bmp'}
OPT=[(tn,tp) for tn in (-3,-2,-1,0) for tp in (0,1,2,3)]
def seg_tables(e):
    out=[]
    for tn,tp in OPT:
        m=((e>=tn)&(e<=tp)).astype(np.int64)
        c=np.where(m==1,e*e+e+0.5,np.where(e>tp,(tp+1)**2,tn*tn)).astype(float)
        out.append((np.cumsum(m),np.cumsum(c)))
    return out
def best(e,L,bounds):
    # bounds: list of boundary tuples (rank splits); returns min cost over thresholds for each class
    best=(1e18,None)
    for b in bounds:
        edges=[0]+list(b)+[len(e)]
        segs=[seg_tables(e[edges[k]:edges[k+1]]) for k in range(len(edges)-1)]
        for combo in itertools.product(range(len(OPT)),repeat=len(segs)):
            cap=0;cost=0.0;ok=False
            for k,o in enumerate(combo):
                cc,cs=segs[k][o]
                if cc.size==0: continue
                if cap+cc[-1]>=L:
                    idx=np.searchsorted(cc,L-cap)  # first index where cum>=L-cap
                    cost+=cs[idx]; ok=True; break
                cap+=cc[-1]; cost+=cs[-1]
            if ok and cost<best[0]: best=(cost,(b,combo))
    return best
if __name__=='__main__':
    for nm,p in imgs.items():
        g=cv2.cvtColor(cv2.resize(cv2.imread(p),(512,512),interpolation=cv2.INTER_CUBIC),cv2.COLOR_BGR2GRAY).astype(np.float64)
        num=0; pn=X.predict(g,num,False)
        so=diag.order(g,num,enc)
        e=np.array([g[i,j]-pn[i,j] for (i,j) in so]).astype(int)
        for payload in (10000,50000,100000):
            L=payload//2+330
            c1=best(e,L,[()])
            bds=[(a,) for a in (8000,16000,32000,48000)]
            c2=best(e,L,bds)
            bds3=[(a,b) for a in (8000,16000,32000) for b in (24000,40000,64000) if a<b]
            c3=best(e,L,bds3)
            f=lambda c: 10*np.log10(c1[0]/c[0])
            print(nm,payload,'global',round(c1[0]),c1[1],'| 2cls gain %.2f dB'%f(c2),c2[1],'| 3cls gain %.2f dB'%f(c3),c3[1],flush=True)
