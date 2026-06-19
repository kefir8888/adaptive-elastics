"""LimX W1 wheeled quadruped: 20 s straight roll through 4 body-height phases (high -> low),
with an ON-THE-FLY adaptive knee (calf/KFE) preload spring (the AdaptivePreloadController
used for the Go1 dog). Phase 1 is tallest (legs most EXTENDED, knees straight); the body
descends through 4 ride heights while rolling forward; the adaptive preload tracks the
(per-leg, measured) knee holding torque.

Renders a video + plots: (a) CoM height vs time, (b) knee electrical power vs time
(no-spring vs adaptive), (c) spring (preload) torque vs time.

  python scripts/limx_phases.py [--video out.mp4]

PHYSICS: on the W1, EXTENDED legs -> HIGH body & LOW knee torque; BENT legs -> LOW body &
HIGH knee torque (standing on straight legs is easy, on bent legs hard). So as the dog
descends through the phases the knee load RISES and the adaptive preload ramps up to track
it (≈4 N*m extended -> ≈26 N*m crouched), holding knee electrical power near zero.
"""
import argparse
import dataclasses
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import limx_roll as L
from pea import energy as E, render_util
from pea.control import AdaptivePreloadController

LEGS, G = L.LEGS, L.G


@dataclasses.dataclass
class Cfg(L.Cfg):
    settle_s: float = 2.0
    drive_s: float = 20.0          # 4 x 5 s phases
    trans_s: float = 1.0           # smooth transition between height phases
    # 4 ride heights by leg-EXTENSION fraction (1.0 = tallest/most-extended, knees straight;
    # lower = more bent/lower body). Phase 1 highest, descending. (Corrected: on the W1,
    # EXTENDED legs -> HIGH body & LOW knee torque; BENT -> LOW body & HIGH knee torque.)
    phase_fracs: tuple = (0.95, 0.68, 0.43, 0.20)
    adapt_window_s: float = 1.5
    adapt_rate: float = 15.0
    adapt_kp: float = 0.5
    tau0_max: float = 45.0


def smoothstep(f):
    f = np.clip(f, 0.0, 1.0)
    return 3 * f * f - 2 * f * f * f


def extension_stances(m, fracs):
    """Per-leg stances at target leg-EXTENSION fractions (1.0 = max downward reach = tallest
    body, knees most extended). Wheel kept under the hip. Returns a list of stance dicts."""
    d = mujoco.MjData(m); ld = {}
    for leg in LEGS:
        rows = []
        for hfe in np.linspace(-2.2, 1.2, 57):
            for kfe in np.linspace(0.0, 2.0, 41):
                d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 0.8
                d.qpos[L.jqp(m, leg + "_HAA")] = 0.0
                d.qpos[L.jqp(m, leg + "_HFE")] = hfe; d.qpos[L.jqp(m, leg + "_KFE")] = kfe
                mujoco.mj_forward(m, d)
                r = L.bpos(m, d, leg + "_wheel") - L.bpos(m, d, leg + "_hip")
                if abs(r[0]) < 0.04:
                    rows.append((-r[2], hfe, kfe))      # downward extension, config
        ld[leg] = sorted(rows)
    out = []
    for frac in fracs:
        st = {}
        for leg in LEGS:
            exts = [x[0] for x in ld[leg]]
            tgt = min(exts) + (max(exts) - min(exts)) * frac
            r = min(ld[leg], key=lambda x: abs(x[0] - tgt)); st[leg] = (0.0, r[1], r[2])
        out.append(st)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", default=None)
    args = ap.parse_args(); cfg = Cfg()
    m = L.build_model(cfg, with_assets=False); d0 = mujoco.MjData(m)
    knee_mc = E.motor_constants(cfg.knee_motor); wheel_mc = E.motor_constants(cfg.wheel_motor)
    Mtot = float(m.body_mass.sum())
    stances = extension_stances(m, cfg.phase_fracs)

    n_settle = int(cfg.settle_s / cfg.dt); n_drive = int(cfg.drive_s / cfg.dt)
    phase_len = cfg.drive_s / 4.0

    def target_stance(t_drive):
        i = min(3, int(t_drive / phase_len)); t_in = t_drive - i * phase_len
        if i < 3 and t_in > phase_len - cfg.trans_s:
            s = smoothstep((t_in - (phase_len - cfg.trans_s)) / cfg.trans_s)
            return {leg: tuple((1 - s) * np.array(stances[i][leg]) + s * np.array(stances[i + 1][leg]))
                    for leg in LEGS}
        return stances[i]

    def com_z(d): return float((d.xipos[1:, 2] * m.body_mass[1:]).sum() / m.body_mass[1:].sum())

    kfe_dof = {leg: L.jdof(m, leg + "_KFE") for leg in LEGS}
    whl_dof = {leg: L.jdof(m, leg + "_WHL") for leg in LEGS}
    kfe_act = {leg: L.aid(m, leg + "_KFE_p") for leg in LEGS}
    tau_roll = cfg.crr * (Mtot * G / 4.0) * cfg.wheel_radius

    def roll(adaptive, frames=None):
        d = mujoco.MjData(m)
        st0 = stances[0]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 1.0
        for leg in LEGS:
            ha, hf, kf = st0[leg]
            d.qpos[L.jqp(m, leg + "_HAA")] = ha; d.qpos[L.jqp(m, leg + "_HFE")] = hf; d.qpos[L.jqp(m, leg + "_KFE")] = kf
        mujoco.mj_forward(m, d)
        wz = min(L.bpos(m, d, leg + "_wheel")[2] for leg in LEGS); d.qpos[2] = 1.0 - (wz - cfg.wheel_radius)
        ctrl = AdaptivePreloadController(4, cfg.dt, cfg.tau0_max, window_s=cfg.adapt_window_s,
                                         kp=cfg.adapt_kp, rate=cfg.adapt_rate) if adaptive else None
        log = {k: [] for k in ("t", "comz", "kneeP", "preload", "kneeTau")}
        for i in range(n_settle + n_drive):
            drive = i >= n_settle
            t_drive = (i - n_settle) * cfg.dt if drive else 0.0
            st = target_stance(t_drive) if drive else st0
            for leg in LEGS:
                ha, hf, kf = st[leg]
                d.ctrl[L.aid(m, leg + "_HAA_p")] = ha; d.ctrl[L.aid(m, leg + "_HFE_p")] = hf
                d.ctrl[L.aid(m, leg + "_KFE_p")] = kf
                d.ctrl[L.aid(m, leg + "_WHL_v")] = cfg.wheel_speed if drive else 0.0
            d.qfrc_applied[:] = 0.0
            tau0 = ctrl.tau0 if adaptive else np.zeros(4)
            for j, leg in enumerate(LEGS):
                d.qfrc_applied[whl_dof[leg]] = -np.sign(d.qvel[whl_dof[leg]]) * tau_roll
                d.qfrc_applied[kfe_dof[leg]] = tau0[j]
            mujoco.mj_step(m, d)
            kt = np.array([float(d.actuator_force[kfe_act[leg]]) for leg in LEGS])
            if adaptive:
                ctrl.update(kt)
            if drive:
                kp = sum(float(E.electrical_power(kt[j], d.qvel[kfe_dof[leg]], knee_mc.kt, knee_mc.r, regen=cfg.regen))
                         for j, leg in enumerate(LEGS))
                log["t"].append(t_drive); log["comz"].append(com_z(d))
                log["kneeP"].append(kp); log["preload"].append(tau0.copy()); log["kneeTau"].append(kt.copy())
                if frames is not None and (i - n_settle) % frames["stride"] == 0:
                    frames["cam"].lookat[:] = d.qpos[:3]
                    frames["renderer"].update_scene(d, camera=frames["cam"])
                    fr = frames["renderer"].render().copy()
                    lines = ["LimX W1  -  rolling through 4 body-height phases (adaptive knee spring)",
                             f"CoM height {com_z(d):.2f} m",
                             f"knee power {kp:4.0f} W   spring {np.mean(tau0):4.0f} N*m"]
                    frames["buf"].append(render_util.overlay_text(fr, lines, size=20,
                        colors=[(235,235,235),(150,220,255),(120,235,140)]))
        for k in log: log[k] = np.array(log[k])
        return log

    base = roll(adaptive=False)
    adpt = roll(adaptive=True)
    print(f"=== LimX W1 4-phase height roll (22 s, total {Mtot:.1f} kg) ===")
    print(f"CoM height range {adpt['comz'].min():.3f}-{adpt['comz'].max():.3f} m across 4 phases")
    print(f"knee electrical power: no-spring mean {base['kneeP'].mean():.0f} W -> adaptive {adpt['kneeP'].mean():.0f} W "
          f"({100*(adpt['kneeP'].mean()-base['kneeP'].mean())/base['kneeP'].mean():+.0f}%)")
    print(f"adaptive preload converged to ~{adpt['preload'][-1].round(1)} N*m per leg")

    # ---- 3-panel time plot ----
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(adpt["t"], adpt["comz"], color="tab:purple"); ax[0].set_ylabel("CoM height (m)")
    ax[0].set_title("LimX W1 — 20 s roll through 4 body-height phases + adaptive knee spring")
    ax[1].plot(base["t"], base["kneeP"], color="tab:red", lw=1, label="no spring")
    ax[1].plot(adpt["t"], adpt["kneeP"], color="tab:green", lw=1, label="adaptive spring")
    ax[1].set_ylabel("knee power\n(4 knees, W)"); ax[1].legend(fontsize=8)
    for j, leg in enumerate(LEGS):
        ax[2].plot(adpt["t"], adpt["preload"][:, j], lw=1, label=f"{leg}")
    ax[2].set_ylabel("spring torque\n(preload, N*m)"); ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=7, ncol=4)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout()
    out_dir = pathlib.Path("outputs/gravity_compensation/raw/limx_phases"); out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "phases_timeplots.png", dpi=130); plt.close(fig)
    print(f"WROTE {out_dir}/phases_timeplots.png")

    if args.video:
        W, H = 960, 540; mv = L.build_model(cfg, with_assets=True); render_util.set_quality(mv, W, H)
        m = mv   # rebind: nested roll() reads `m` via closure (joints/actuators are identical
                 # on the textured model, so the precomputed dof/actuator ids stay valid)
        fr = {"renderer": mujoco.Renderer(mv, H, W), "cam": render_util.free_camera(3.6, -16.0, 120.0),
              "stride": max(1, round((1/30)/cfg.dt)), "buf": []}
        roll(adaptive=True, frames=fr)
        render_util.write_mp4(fr["buf"], args.video, 30, W, H)


if __name__ == "__main__":
    main()
