# run_bench.py —— 批量运行各 strategy，汇总 results/all.csv
# 用法： python run_bench.py [data_dir]
import os, re, subprocess, sys, csv

HERE  = os.path.dirname(os.path.abspath(__file__))
DATA  = sys.argv[1] if len(sys.argv) > 1 else "../lab05/ann/data/"
EXE   = os.path.join(HERE, "main.exe")
MPI   = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"
OUT   = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)

def run(args):
    p = subprocess.run(args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.stdout.decode("utf-8", errors="replace")

rows = []
def add(method, label, rec, us, sp, qps=""):
    rows.append((method, label, f"{float(rec):.4f}", f"{float(us):.1f}",
                 f"{float(sp):.2f}", (f"{float(qps):.0f}" if qps != "" else "")))

# ---- S0 Flat 暴力基线 ----
txt = run([EXE, "0", DATA]); open(f"{OUT}/s0.txt","w",encoding="utf-8").write(txt)
m = re.search(r"= ([\d.]+) us/query \(avx2\)", txt)
base = float(m.group(1)) if m else 1562.5
print("baseline avx2 =", base, "us/q")

# ---- S1 CPU IVF / IVF-PQ ----
txt = run([EXE, "1", DATA, "8"]); open(f"{OUT}/s1.txt","w",encoding="utf-8").write(txt)
for mm in re.finditer(r"^(IVF|IVF-PQ),(\d+),(\d+),(\d+),(\d+),(\d+),([\d.]+),([\d.]+),([\d.]+)", txt, re.M):
    meth, nlist, nprobe, pqm, rk, thr, rec, us, sp = mm.groups()
    lbl = f"nprobe={nprobe}" + (f",rk={rk}" if meth=="IVF-PQ" else "")
    add(f"CPU-{meth}", lbl, rec, us, sp)

# ---- S2 GPU cuBLAS ----
txt = run([EXE, "2", DATA, "512"]); open(f"{OUT}/s2.txt","w",encoding="utf-8").write(txt)
mm = re.search(r"GPU-flat-exact,\d+,\d+,\d+,([\d.]+),([\d.]+),([\d.]+)", txt)
if mm: add("GPU-flat", "exact", *mm.groups())

# ---- S3 LSH - CPU ----
txt = run([EXE, "3", DATA, "8"]); open(f"{OUT}/s3.txt","w",encoding="utf-8").write(txt)
for mm in re.finditer(r"^LSH,(\d+),(\d+),\d+,([\d.]+),([\d.]+),([\d.]+)", txt, re.M):
    K, rk, rec, us, sp = mm.groups()
    add("LSH-CPU", f"K={K},rk={rk}", rec, us, sp)

# ---- S4 LSH - GPU ----
txt = run([EXE, "4", DATA, "512"]); open(f"{OUT}/s4.txt","w",encoding="utf-8").write(txt)
for mm in re.finditer(r"^LSH-GPU,(\d+),(\d+),([\d.]+),([\d.]+),([\d.]+)", txt, re.M):
    K, rk, rec, us, sp = mm.groups()
    add("LSH-GPU", f"K={K},rk={rk}", rec, us, sp)

# ---- S5 异构协同分流：exact + LSH 两种 GPU 分支 ----
for algo, aname in [(0, "exact"), (1, "LSH")]:
    txt = run([EXE, "5", DATA, "512", "8", "16", str(algo), "500"])
    open(f"{OUT}/s5_{aname}.txt","w",encoding="utf-8").write(txt)
    for mm in re.finditer(r"^([01]\.\d+),([\d.]+),([\d.]+),([\d.]+),([\d.]+)", txt, re.M):
        f, rec, us, qps, sp = mm.groups()
        add(f"hetero-{aname}", f"gpu_frac={f}", rec, us, sp, qps)

# ---- S6 MPI + 异构分流：exact + LSH 两种，扫 gpu_frac ----
if os.path.exists(MPI):
    for algo, aname in [(0, "exact"), (1, "LSH")]:
        fracs = ["0.55"] if algo == 0 else ["0.70", "0.725", "0.75", "0.8", "0.85", "0.9", "0.95", "1.0"]
        for frac in fracs:
            nproc = 2
            txt = run([MPI, "-n", str(nproc), EXE, "6", DATA, str(algo), frac, "16", "6"])
            open(f"{OUT}/s6_{aname}_n{nproc}_{frac}.txt","w",encoding="utf-8").write(txt)
            mm = re.search(r"recall=([\d.]+)\s+makespan=[\d.]+ ms\s+us/query=([\d.]+)\s+qps=([\d.]+)\s+speedup=([\d.]+)", txt)
            if mm:
                rec, us, qps, sp = mm.groups()
                add(f"MPI-{aname}", f"n={nproc},frac={frac}", rec, us, sp, qps)
else:
    print("[skip] mpiexec 不存在, 跳过 S6")

with open(f"{OUT}/all.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["method","label","recall","us_per_query","speedup","qps"])
    w.writerows(rows)
print(f"[done] {len(rows)} rows -> {OUT}/all.csv   baseline={base} us/q")
