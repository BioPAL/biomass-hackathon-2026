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
    sol = invert_closure(phi_obs, trip, T)
    d_est, d_true = sol["m"] - sol["m"][0], m_true - m_true[0]
    print("dSM true:", d_true.round(3)); print("dSM est :", d_est.round(3)); print("rms resid rad:", round(sol["residual_rms_rad"], 3))
    assert np.sqrt(np.mean((d_est - d_true) ** 2)) < 0.05, "relative moisture not recovered"
    sol2 = invert_closure(phi_obs, trip, T, coh_obs=coh_obs, pairs=pairs, w_coh=0.5)
    print("abs SM with coherence magnitude:", sol2["m"].round(3), "true:", m_true)

if __name__ == "__main__":
    test_recovers_drydown()
