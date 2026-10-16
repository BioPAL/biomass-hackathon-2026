"""Synthetic end-to-end check of the closure-phase estimator: forward model -> noise -> inversion recovers the dry-down."""
import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sm_pipeline.estimators import closure_model, invert_closure, coherence_model
from sm_pipeline.observables import all_triplets

def test_recovers_drydown():
    m_true = np.array([0.31, 0.27, 0.22, 0.17, 0.14, 0.12, 0.11])  # Maydantal-like ERA5-Land dry-down, 7 acquisitions
    T = len(m_true); trip = all_triplets(T); pairs = [(i, j) for i in range(T) for j in range(i + 1, T)]
    phi = closure_model(m_true, trip); rng = np.random.default_rng(1)
    phi_obs = phi + rng.normal(0, 0.05, phi.shape)                    # 0.05 rad closure noise after multilooking
    coh_obs = np.array([abs(coherence_model(m_true[i], m_true[j])) for i, j in pairs]) * 0.9
    print("closure phases (deg), consecutive triplets:", np.degrees(phi[[trip.index((i, i + 1, i + 2)) for i in range(T - 2)]]).round(1))
    d_true = m_true - m_true[0]; print("dSM true:", d_true.round(3))
    # 1) implementation check (must pass): without noise the inversion returns the dry-down
    sol0 = invert_closure(phi, trip, T); d0 = sol0["m"] - sol0["m"][0]; rmse0 = float(np.sqrt(np.mean((d0 - d_true) ** 2)))
    print("dSM est (no noise)       :", d0.round(3), "rmse", round(rmse0, 3))
    assert rmse0 < 0.02, f"inversion does not recover the noise-free dry-down (rmse {rmse0:.3f} m3/m3)"
    # 2) method property (reported, not asserted): 0.05 rad closure noise -> about 0.08 m3/m3 rmse (README, known limitations)
    sol = invert_closure(phi_obs, trip, T); d_est = sol["m"] - sol["m"][0]; rmse = float(np.sqrt(np.mean((d_est - d_true) ** 2)))
    print("dSM est (0.05 rad noise) :", d_est.round(3), "rmse", round(rmse, 3), "| rms resid rad:", round(sol["residual_rms_rad"], 3))
    sol2 = invert_closure(phi_obs, trip, T, coh_obs=coh_obs, pairs=pairs, w_coh=0.5)
    print("abs SM with coherence magnitude:", sol2["m"].round(3), "true:", m_true)

if __name__ == "__main__":
    test_recovers_drydown(); print("PASS closure inversion (noise-free); noisy case reported above")
