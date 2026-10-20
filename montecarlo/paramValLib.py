import os
import re
import subprocess
import tempfile

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NGSPICE_BIN = os.path.expanduser("~/.conda/envs/spice_env/bin/ngspice")
LIB_PATH = (
    "/home/oliviag/skywater130nm/volare/sky130/versions/"
    "a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/"
    "libs.tech/ngspice/sky130.lib.spice"
)
CORNER = "tt_77k"

TEMP_K = 77.0
TEMP_C = TEMP_K - 273.15

PARAMS = ["vth0", "u0", "rdsw", "nfactor", "vsat", "eta0", "delta"]
ONE = {p: 1.0 for p in PARAMS}

# Bisection search window and resolution
LO_START, HI_START = -10.0, 10.0
TOL = 1e-3
MAX_ITERS = 40

DEBUG = False

VGS_SWEEP = [round(0.05 * i, 3) for i in range(19)]  # 0 -> 0.90 V

# Substrings that indicate an ngspice failure
BAD = [
    "fatal", "simulation(s) aborted", "singular matrix", "no convergence",
    "iteration limit reached", "error on line", "no such parameter",
    "unknown subckt", "could not find", "can't find",
]

# --------------------------------------------------------------------------- #
# Deck template
# --------------------------------------------------------------------------- #
DECK = """\
* BSIM4 {tk:.0f}K multiplier test
.lib "{lib}" {corner}
.temp {tc}

XM1 d g 0 0 sky130_fd_pr__nfet_01v8_lvt l=0.15 w=1.6 nf=1
+ ad='floor((nf+1)/2) * w/nf * 0.29' as='floor((nf+2)/2) * w/nf * 0.29'
+ pd='2 * floor((nf+1)/2) * (w/nf + 0.29)' ps='2 * floor((nf+2)/2) * (w/nf + 0.29)'
+ nrd='0.29 / w' nrs='0.29 / w' sa=0 sb=0 sd=0

vd d 0 0.9
vg g 0 {vg}

.control
{alters}reset
{body}
quit
.endc
.end
"""


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def read_base_value(param, text):
    """Parse ``param = <number>`` out of a model-card text blob."""
    m = re.search(rf"\b{re.escape(param)}\s*=\s*([-+0-9.eE]+)", text)
    if not m:
        raise ValueError(f"Could not find '{param}' in model file")
    return float(m.group(1))


def build_deck(mults, vg, body):
    """Render the deck for a given multiplier dict, gate voltage, and control body."""
    alters = "".join(f"alterparam {p}_nom = {mults[p]:.10g}\n" for p in PARAMS)
    return DECK.format(
        tk=TEMP_K, lib=LIB_PATH, corner=CORNER,
        tc=TEMP_C, vg=vg, alters=alters, body=body,
    )


def _run(text, d, fname="test.cir"):
    """Write ``text`` to ``d/fname``, run ngspice -b, return (ok, combined_output)."""
    path = os.path.join(d, fname)
    with open(path, "w") as f:
        f.write(text)
    try:
        pr = subprocess.run(
            [NGSPICE_BIN, "-b", path],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    out = pr.stdout + pr.stderr
    if DEBUG:
        print(f"--- ngspice rc={pr.returncode} ---\n{out}\n")
    ok = pr.returncode == 0 and not any(b in out.lower() for b in BAD)
    return ok, out


# --------------------------------------------------------------------------- #
# Simulation runs
# --------------------------------------------------------------------------- #
def run_op(mults):
    """Run an operating-point sim with the given multipliers. Returns (ok, output)."""
    with tempfile.TemporaryDirectory() as d:
        return _run(build_deck(mults, "0.9", "op"), d)


def run_iv(mults):
    """Run an Id-Vgs sweep. Returns (ok, [(vgs, id), ...]) or (False, None)."""
    with tempfile.TemporaryDirectory() as d:
        csv = os.path.join(d, "iv.csv")
        body = f"dc vg 0 0.9 0.05\nwrdata {csv} i(vd)"
        ok, out = _run(build_deck(mults, "0", body), d, "iv.cir")
        if not ok or not os.path.isfile(csv):
            return False, None
        curve = []
        with open(csv) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        curve.append((float(parts[0]), -float(parts[1])))
                    except ValueError:
                        continue
    return (len(curve) > 1), (curve or None)


# --------------------------------------------------------------------------- #
# Multiplier testing / bound finding
# --------------------------------------------------------------------------- #
def test_mults(mults):
    """True if the OP sim passes for the given multiplier dict."""
    return run_op(mults)[0]


def test_multiplier(param, mult):
    """True if setting only ``param`` to ``mult`` (others = 1.0) still passes."""
    return test_mults({p: (mult if p == param else 1.0) for p in PARAMS})


def bisect_bound(param, good, bad):
    """Bisect between a passing (``good``) and failing (``bad``) multiplier."""
    for _ in range(MAX_ITERS):
        if abs(good - bad) < TOL:
            break
        mid = (good + bad) / 2.0
        if test_multiplier(param, mid):
            good = mid
        else:
            bad = mid
    return good


def find_bounds(param):
    """Return (lower, upper, lower_note, upper_note) multiplier bounds for ``param``."""
    if test_multiplier(param, HI_START):
        upper, unote = HI_START, f">= {HI_START} (no failure in bracket)"
    else:
        upper = bisect_bound(param, 1.0, HI_START)
        unote = f"{upper:.4f}"
    if test_multiplier(param, LO_START):
        lower, lnote = LO_START, f"<= {LO_START} (no failure in bracket)"
    else:
        lower = bisect_bound(param, 1.0, LO_START)
        lnote = f"{lower:.4f}"
    return lower, upper, lnote, unote


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main():
    import sys

    print(NGSPICE_BIN)
    print(f"Target temp: {TEMP_K} K ({TEMP_C:.2f} C)")

    ok, out = run_op(ONE)
    print(f"Baseline (all _nom = 1) at {TEMP_K}K: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print(out)
        sys.exit("Baseline fails before any multiplier is applied -- fix setup first.")

    _, c1 = run_iv(ONE)
    _, c2 = run_iv({**ONE, "u0": 0.5})
    if c1 and c2 and c1 == c2:
        sys.exit("u0_nom = 0.5 gave an identical curve -- alterparam is a no-op. "
                 "Check the _nom hooks in the model card.")
    print("alterparam hooks confirmed live (u0_nom changed the IV curve).")

    print(f"\n{'param':10s} {'min mult':>14s} {'max mult':>14s}")
    print("-" * 42)
    results = {}
    for p in PARAMS:
        lower, upper, lnote, unote = find_bounds(p)
        results[p] = (lower, upper)
        print(f"{p:10s} {lnote:>14s} {unote:>14s}")

    print("\nNOTE: per-parameter bounds with the other 6 held at 1.0. A joint")
    print("draw can still fail (esp. u0/vsat/rdsw interactions).")
    return results


if __name__ == "__main__":
    main()
