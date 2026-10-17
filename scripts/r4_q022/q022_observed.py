"""Q-022 (SCRATCH): observed transfer ratio E_obs(s)/T for a finite 'true' target (observation only)."""
import numpy as np, importlib.util, sys
spec = importlib.util.spec_from_file_location("tr", str(__import__("pathlib").Path(__file__).with_name("q022_transfer.py"))); tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
M1, M2, S = 101, 1601, 40
for r in (1.0, 0.2, 5.0):
    P2 = tr.setup(M2, r); P1 = tr.setup(M1, r); K1 = (M1 + 1) // 2
    tau = P2['om'].copy(); tau[:K1, :K1] = 0; T = tr.enorm(P2, tau)
    for N in (2, 3):
        f2, _ = tr.run(P2, N, S); f1, _ = tr.run(P1, N, S)
        obs = []
        for s in range(1, S + 1):
            Z = np.zeros_like(P2['om']); Z[:K1, :K1] = f1[s - 1]; obs.append(tr.enorm(P2, f2[s - 1] - Z) / T)
        print("r=%.1f N=%d  E_obs/T at s=1,2,3,4,6,10,20,40: %s" % (r, N, " ".join("%.3f" % obs[s - 1] for s in (1, 2, 3, 4, 6, 10, 20, 40))))
