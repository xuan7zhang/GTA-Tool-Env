"""tab:tgb-agreement — mask overlap between acc-based and likelihood selection.

Per model: LOO tau* per-task masks (tau* = argmax train acc in the stored
sweep) vs Algorithm 1 masks at both levels:
  - task level (star_of, per-task; the well-posedness probe)
  - split level (star, one global mask)
plus the naive likelihood-Shapley subset (phi^L>0) where tgb_shapley exists.
Reports mean Jaccard over test tasks, per-(task,tool) decision agreement, and
each method's held-out accuracy as stored by its own artifact.

  python -m tgb.agreement_v4
"""
import json
import os
import statistics as st

D = "/datasets/omni_pretraining/gta2/results/taco/tgb4"
TAGS = ["7b", "14b", "llama8b", "mistral7b"]
MENU = None


def tau_star(loo):
    sw = {k: v for k, v in loo["sweep"].items() if k != "S"}
    return max(sw, key=lambda k: sw[k]["train"])


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def agree(a, b, menu):
    a, b = set(a), set(b)
    return sum(1 for t in menu if (t in a) == (t in b)) / len(menu)


def main():
    global MENU
    import sys
    suf = "_self_S" if "--self" in sys.argv else ""
    for tag in TAGS:
        loo_p = f"{D}/tgb_loo_{tag}{suf}.json"
        if not os.path.exists(loo_p):
            print(f"[{tag}] no LOO artifact, skip"); continue
        loo = json.load(open(loo_p))
        MENU = loo["S"]
        ts = tau_star(loo)
        row = loo["sweep"][ts]
        masks_loo = row["masks"]
        print(f"\n[{tag}] LOO tau*={ts}  test acc {row['test']:.3f}  "
              f"tools {row['tools']:.2f}  ({len(masks_loo)} task masks)")

        for base in ("union", "empty"):
            p = f"{D}/tgb_algo1_{tag}_task_{base}.json"
            if not os.path.exists(p):
                print(f"  algo1 task/{base}: missing"); continue
            al = json.load(open(p))
            if "star_of" not in al:
                print(f"  algo1 task/{base}: no star_of (old run)"); continue
            star = al["star_of"]
            common = [t for t in star if t in masks_loo]
            J = [jac(masks_loo[t], star[t]) for t in common]
            A = [agree(masks_loo[t], star[t], MENU) for t in common]
            print(f"  vs algo1 task/{base:<6} n={len(common)}  "
                  f"Jaccard {st.mean(J):.3f}  per-tool agree {st.mean(A):.3f}  "
                  f"algo1 acc {al['test_acc']['algo1']:.3f}  "
                  f"tools {al['tools']:.2f}")

        for base in ("union", "empty"):
            p = f"{D}/tgb_algo1_{tag}_split_{base}.json"
            if not os.path.exists(p):
                continue
            al = json.load(open(p))
            g = al["star"]
            J = [jac(masks_loo[t], g) for t in masks_loo]
            A = [agree(masks_loo[t], g, MENU) for t in masks_loo]
            print(f"  vs algo1 split/{base:<6} (global {len(g)} tools)  "
                  f"Jaccard {st.mean(J):.3f}  per-tool agree {st.mean(A):.3f}  "
                  f"algo1 acc {al['test_acc']['algo1']:.3f}")

        sp = f"{D}/tgb_shapley_{tag}.json"
        if os.path.exists(sp):
            sh = json.load(open(sp))
            picked = sorted(t for t, v in sh["phiL"].items() if v > 0) or \
                sorted(sh["menu"])
            key = ",".join(picked)
            acc = sh["heldout"].get(key)
            J = [jac(masks_loo[t], picked) for t in masks_loo]
            print(f"  naive phi^L>0 subset {picked}  "
                  f"held-out {acc if acc is None else round(acc, 3)}  "
                  f"Jaccard vs LOO {st.mean(J):.3f}")


if __name__ == "__main__":
    main()
