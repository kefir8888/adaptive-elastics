"""Thin shim onto pea.experiment (the robot x task x spring sweep harness).

Examples:
  # G1: stiffness sweep at the hip-pitch across three tasks
  pea-sweep --run <run> --tasks stand:0,0,0 walk:1,0,0 fast:1.5,0,0 \
            --joint hip_pitch --k 0 40 68 100 --theta0 -0.29 --out sweep.csv

  # different robot constants / joint
  pea-sweep --run <go2_run> --robot go2 --joint knee --k 50 100 150
"""

from pea.experiment import main

if __name__ == "__main__":
    main()
