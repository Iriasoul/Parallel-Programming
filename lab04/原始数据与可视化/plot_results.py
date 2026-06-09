import csv, argparse, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- 读入并按 (strategy,P,omp,part,nprobe,ef,topk) 全局取最小延迟 ----------
def load(csvs):
    g = defaultdict(lambda: (1e18, 0.0))
    for f in csvs:
        if not os.path.exists(f): 
            print(f"  [skip] {f} 不存在"); continue
        for r in csv.DictReader(open(f)):
            gk = (int(r['strategy']), int(r['P']), int(r['omp']), r['part'],
                  int(r['nprobe']), int(r['ef']), int(r['topk']))
            lat, rec = float(r['latency_us']), float(r['recall'])
            if lat < g[gk][0]:
                g[gk] = (lat, rec)
    return g

def Q(g, s, P, omp=4, part='block', nprobe=16, ef=64, topk=10):
    """取一条记录 (lat, rec); 取不到返回 None"""
    return g.get((s, P, omp, part, nprobe, ef, topk))

# 配色
C = {'ivf':'#2c6fbb', 'reduce':'#e08214', 'batch':'#1b7837', 'overlap':'#762a83',
     'pq1':'#d6604d', 'pq2':'#b2182b', 'hnsw':'#5aae61', 'gray':'#888888'}
plt.rcParams.update({'font.size': 11, 'figure.dpi': 130, 'axes.grid': True,
                     'grid.alpha': 0.3, 'savefig.bbox': 'tight'})

# ---------------- 图1: 各策略最优加速比柱状 ----------------
def fig_speedup(g, out):
    S0 = Q(g,0,1)[0]
    # (label, lat) 取每个策略的代表配置(与报告总览表一致)
    items = [
        ('S0',  Q(g,0,1)[0]),
        ('S1',  Q(g,1,4)[0]),  ('S2', Q(g,2,4,part='cyclic')[0]),
        ('S3',  Q(g,3,4)[0]),  ('S4', Q(g,4,2,omp=2)[0]),
        ('S5',  Q(g,5,8)[0]),  ('S6', Q(g,6,8)[0]),
        ('S7',  Q(g,7,4)[0]),  ('S8', Q(g,8,2,omp=4)[0]),
        ('S9',  Q(g,9,4)[0]),  ('S10',Q(g,10,2,omp=4)[0]),
        ('S11', Q(g,11,1)[0]), ('S12',Q(g,12,2,omp=4)[0]),
        ('S13', Q(g,13,4)[0]), ('S14',Q(g,14,4)[0]),
    ]
    labels = [a for a,_ in items]
    sp = [S0/b for _,b in items]
    grp = (['gray']+['ivf']*6+['pq1']*4+['hnsw']*4)
    colors = [C[k] for k in grp]
    fig, ax = plt.subplots(figsize=(9,4.2))
    bars = ax.bar(labels, sp, color=colors)
    ax.axhline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    ax.set_ylabel('Speedup (vs S0)'); ax.set_title('Speedup of each strategy (best config)')
    for b,v in zip(bars,sp):
        ax.text(b.get_x()+b.get_width()/2, v+0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, max(sp)*1.15)
    fig.savefig(f'{out}/chart_speedup.png'); plt.close(fig)

# ---------------- 图2: MPI-IVF 四通信策略 延迟随 P 折线 ----------------
def fig_scaling(g, out):
    Ps=[1,2,4,8]
    series=[('S1 Gather','ivf',1),('S3 Reduce','reduce',3),
            ('S5 Batch','batch',5),('S6 Overlap','overlap',6)]
    fig,ax=plt.subplots(figsize=(7,4.5))
    for name,c,s in series:
        ys=[Q(g,s,P)[0] for P in Ps]
        ax.plot(Ps,ys,'-o',color=C[c],label=name,lw=1.8,ms=6)
        for P,y in zip(Ps,ys): ax.text(P,y+8,f'{y:.0f}',ha='center',fontsize=8,color=C[c])
    ax.set_xticks(Ps); ax.set_xlabel('Processes P'); ax.set_ylabel('Latency (us)')
    ax.set_title('MPI-IVF latency vs P (communication strategies)')
    ax.legend(); fig.savefig(f'{out}/chart_scaling.png'); plt.close(fig)

# ---------------- 图3: S4 P×T 配比柱状 ----------------
def fig_hybrid(g, out):
    cfg=[('1x1',1,1),('1x2',1,2),('1x4',1,4),('1x8',1,8),('2x2',2,2),('2x4',2,4),('4x2',4,2)]
    labels=[c[0] for c in cfg]
    ys=[Q(g,4,P,omp=T)[0] for _,P,T in cfg]
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    bars=ax.bar(labels,ys,color=C['ivf'])
    best=min(range(len(ys)),key=lambda i:ys[i])
    bars[best].set_color(C['batch'])
    for b,v in zip(bars,ys): ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
    ax.set_xlabel('P x T'); ax.set_ylabel('Latency (us)')
    ax.set_title('MPI x OpenMP hybrid (S4) latency by P x T')
    ax.set_ylim(0,max(ys)*1.12)
    fig.savefig(f'{out}/chart_hybrid.png'); plt.close(fig)

# ---------------- 图4: IVF/PQ recall-latency 散点(nprobe扫描) ----------------
def fig_nprobe(g, out):
    nps=[4,8,16,32,64]
    series=[('IVF (S1)','ivf',1),('IVF-PQ M1 (S7)','pq1',7),('IVF-PQ M2 (S9)','pq2',9)]
    fig,ax=plt.subplots(figsize=(7,5))
    for name,c,s in series:
        pts=[Q(g,s,1,nprobe=n) for n in nps]
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        ax.plot(xs,ys,'-o',color=C[c],label=name,lw=1.6,ms=6)
        for n,x,y in zip(nps,xs,ys): ax.text(x,y-0.006,f'{n}',ha='center',fontsize=7,color=C[c])
    ax.set_xlabel('Latency (us)'); ax.set_ylabel('Recall@10')
    ax.set_title('IVF / IVF-PQ recall-latency (nprobe sweep, P=1)')
    ax.legend(loc='lower right'); fig.savefig(f'{out}/chart_nprobe_scatter.png'); plt.close(fig)

# ---------------- 图5: HNSW recall-latency 散点(ef扫描) ----------------
def fig_ef(g, out):
    efs=[16,32,64,128,200]
    pts=[Q(g,11,1,ef=e) for e in efs]
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(xs,ys,'-o',color=C['hnsw'],lw=1.8,ms=7)
    for e,x,y in zip(efs,xs,ys): ax.text(x+8,y-0.004,f'ef={e}',fontsize=8,color=C['hnsw'])
    ax.set_xlabel('Latency (us)'); ax.set_ylabel('Recall@10')
    ax.set_title('HNSW recall-latency (ef_search sweep, S11 P=1)')
    fig.savefig(f'{out}/chart_ef_scatter.png'); plt.close(fig)

# ---------------- 图6: 增大P 与 增大nprobe 两条召回提升路径 ----------------
def fig_PvsNprobe(g, out):
    # 路径A: P=1 固定, nprobe=4..64 (延迟随精度上升)
    nps=[4,8,16,32,64]
    A=[Q(g,1,1,nprobe=n) for n in nps]
    Ax=[p[0] for p in A]; Ay=[p[1] for p in A]
    # 路径B: nprobe=16 固定, P=1..8 (延迟随并行下降, recall 随广度上升)
    Ps=[1,2,4,8]
    B=[Q(g,1,P,nprobe=16) for P in Ps]
    Bx=[p[0] for p in B]; By=[p[1] for p in B]
    fig,ax=plt.subplots(figsize=(7.5,5))
    ax.plot(Ax,Ay,'-o',color=C['ivf'],lw=1.8,ms=6,label='Increase nprobe (fix P=1)')
    for n,x,y in zip(nps,Ax,Ay): ax.text(x,y-0.007,f'np={n}',ha='center',fontsize=7,color=C['ivf'])
    ax.plot(Bx,By,'-s',color=C['batch'],lw=1.8,ms=6,label='Increase P (fix nprobe=16)')
    for P,x,y in zip(Ps,Bx,By): ax.text(x,y+0.004,f'P={P}',ha='center',fontsize=7,color=C['batch'])
    ax.set_xlabel('Latency (us)'); ax.set_ylabel('Recall@10')
    ax.set_title('Two paths to higher recall: more nprobe vs more processes')
    ax.legend(loc='lower right'); fig.savefig(f'{out}/chart_PvsNprobe.png'); plt.close(fig)

# ---------------- 图7: 数据并行 vs 任务并行 (左:召回 右:延迟) ----------------
def fig_dvt(g, out):
    Ps=[1,2,4,8]
    s11r=[Q(g,11,P)[1] for P in Ps]; s11l=[Q(g,11,P)[0] for P in Ps]
    s13r=[Q(g,13,P)[1] for P in Ps]; s13l=[Q(g,13,P)[0] for P in Ps]
    s14r=[Q(g,14,P)[1] for P in Ps]; s14l=[Q(g,14,P)[0] for P in Ps]
    sty=[('S11 multi-block (data-parallel)', C['hnsw'],'-o'),
         ('S13 IVF+HNSW (task-parallel)',    C['pq1'], '-s'),
         ('S14 HNSW-on-HNSW (task-parallel)',C['reduce'],'-^')]
    fig,(axL,axR)=plt.subplots(1,2,figsize=(12,4.6))
    # 左: 召回率
    for (name,c,m),y in zip(sty,[s11r,s13r,s14r]):
        axL.plot(Ps,y,m,color=c,lw=1.8,ms=6,label=name)
    axL.set_xticks(Ps); axL.set_xlabel('Processes P'); axL.set_ylabel('Recall@10')
    axL.set_title('(a) Recall vs P')
    # 右: 平均延迟
    for (name,c,m),y in zip(sty,[s11l,s13l,s14l]):
        axR.plot(Ps,y,m,color=c,lw=1.8,ms=6,label=name)
    axR.set_xticks(Ps); axR.set_xlabel('Processes P'); axR.set_ylabel('Latency (us)')
    axR.set_title('(b) Latency vs P')
    # 共用图例(放底部)
    h,l=axL.get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.04),frameon=False)
    fig.suptitle('Data-parallel vs Task-parallel',
                 fontsize=11)
    fig.tight_layout(rect=[0,0.02,1,0.97])
    fig.savefig(f'{out}/chart_dvt_recall.png'); plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',nargs='+',
                    default=['result.csv','result2.csv','result3.csv'])
    ap.add_argument('--out',default='figs')
    a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    g=load(a.csv)
    print(f"合并 {len(g)} 个配置, 输出到 {a.out}/")
    fig_speedup(g,a.out);    print("  chart_speedup.png")
    fig_scaling(g,a.out);    print("  chart_scaling.png")
    fig_hybrid(g,a.out);     print("  chart_hybrid.png")
    fig_nprobe(g,a.out);     print("  chart_nprobe_scatter.png")
    fig_ef(g,a.out);         print("  chart_ef_scatter.png")
    fig_PvsNprobe(g,a.out);  print("  chart_PvsNprobe.png")
    fig_dvt(g,a.out);        print("  chart_dvt_recall.png")
    print("完成")

if __name__=='__main__':
    main()