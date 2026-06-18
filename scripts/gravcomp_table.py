"""Combine the Galaxea-lift and LimX-roll results into ONE energy-savings table:
savings in the spring-target motors, and in whole-robot consumption (all servos + a
fixed onboard computer, swept). Reads outputs/{galaxea,limx}/*_results.json.

  python scripts/gravcomp_table.py [--compute 150] [--out outputs/gravcomp_table.md]
"""
import argparse
import json
import pathlib

COMPUTE_SWEEP = [0, 50, 150, 300]
FILES = ["outputs/gravity_compensation/raw/galaxea/galaxea_results.json",
         "outputs/gravity_compensation/raw/limx/limx_results.json"]


def pct(a, b):
    return 100 * (b - a) / a if a else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute", type=float, default=150.0)
    ap.add_argument("--out", default="outputs/gravity_compensation/05_energy_savings_table.md")
    args = ap.parse_args()

    rows = [json.loads(pathlib.Path(f).read_text()) for f in FILES if pathlib.Path(f).exists()]
    L = []
    L.append("# Parallel-elastic gravity compensation — energy savings (no regen)\n")
    L.append("Spring-target-motor saving, and whole-robot saving (all servos + a fixed "
             "onboard computer). Motor constants are sourced estimates; the win is "
             "largely gear-independent.\n")
    L.append("| robot / motion | target motors | target energy base→spring (J) | "
             "target saving | whole-robot saving @150 W computer |")
    L.append("|---|---|---|---|---|")
    for r in rows:
        ob = r["all_servo_base_J"] + args.compute * r["duration_s"]
        os_ = r["all_servo_spring_J"] + args.compute * r["duration_s"]
        L.append(f"| {r['robot']} | {r['target_label']} | "
                 f"{r['target_base_J']:.0f} → {r['target_spring_J']:.0f} | "
                 f"{pct(r['target_base_J'], r['target_spring_J']):+.0f}% | "
                 f"{pct(ob, os_):+.1f}% |")
    L.append("\n## Whole-robot saving vs assumed onboard-computer power\n")
    head = "| robot | " + " | ".join(f"{w} W" for w in COMPUTE_SWEEP) + " |"
    L.append(head); L.append("|" + "---|" * (len(COMPUTE_SWEEP) + 1))
    for r in rows:
        cells = []
        for w in COMPUTE_SWEEP:
            ob = r["all_servo_base_J"] + w * r["duration_s"]
            os_ = r["all_servo_spring_J"] + w * r["duration_s"]
            cells.append(f"{pct(ob, os_):+.0f}%")
        L.append(f"| {r['robot'].split('(')[0].strip()} | " + " | ".join(cells) + " |")
    L.append("\n_All-servo (0 W computer) saving: " +
             ", ".join(f"{r['robot'].split('(')[0].strip()} "
                       f"{pct(r['all_servo_base_J'], r['all_servo_spring_J']):+.0f}%"
                       for r in rows) + "._")
    text = "\n".join(L)
    print(text)
    pathlib.Path(args.out).write_text(text + "\n")
    print(f"\nWROTE {args.out}")


if __name__ == "__main__":
    main()
