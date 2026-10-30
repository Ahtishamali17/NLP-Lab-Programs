"""
Runs every program and stores its console output in outputs/NN_name.txt  (this gives you the "output" part
of the practical file).    Usage:
    python run_all.py              # all 32 programs
    python run_all.py 1 5 12       # only programs 1, 5 and 12
    python run_all.py --skip-heavy # skip the programs that train / download large deep-learning models (22-30)
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)
HEAVY = set(range(22, 31))

scripts = sorted(p for p in ROOT.glob("CO*/[0-9][0-9]_*.py"))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
wanted = {int(a) for a in args}
env = dict(os.environ, PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="2", PYTHONUTF8="1")

summary = []
for s in scripts:
    n = int(s.name[:2])
    if wanted and n not in wanted: continue
    if "--skip-heavy" in sys.argv and n in HEAVY: continue
    print(f"[{n:02d}] running {s.name} ...", end=" ", flush=True)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(s)], capture_output=True, text=True, timeout=3600, env=env, cwd=ROOT)
        ok = r.returncode == 0
        (OUT / f"{s.stem}.txt").write_text(r.stdout + ("\n--- STDERR ---\n" + r.stderr[-3000:] if not ok else ""), encoding="utf-8")
    except subprocess.TimeoutExpired:
        ok = False
    summary.append((n, s.name, ok))
    print("OK" if ok else "FAILED", f"({time.time()-t:.0f}s)")

print("\n==== SUMMARY ====")
for n, name, ok in summary:
    print(f"{n:02d}  {'PASS' if ok else 'FAIL'}  {name}")
print(f"{sum(o for *_, o in summary)}/{len(summary)} programs completed. Outputs are in the outputs/ folder.")
