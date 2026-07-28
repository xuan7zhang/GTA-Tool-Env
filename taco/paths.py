"""Single source of truth for TACO output locations.

The output root is overridable with `TACO_ROOT` so that two independent
sessions can run the same code concurrently without writing over each
other's runs, feature tables or report. The default is unchanged
(`$GTA_BIG/results/taco`), so anything already driving these modules keeps
its existing paths; a session that wants isolation exports TACO_ROOT.

Inside whatever root is chosen, the directory tree required by the TACO spec
(§27) is reproduced verbatim: configs/, splits.json, phase0_audit/,
interventions/, raw_runs/, paired_effects/, feature_tables/, models/,
searches/, frozen_environments/, heldout/, plots/, reports/.
"""
import os

BIG = os.environ.get("GTA_BIG", "/datasets/omni_pretraining/gta2")
TACO = os.environ.get("TACO_ROOT", f"{BIG}/results/taco")

DS_PATH = f"{BIG}/data/gta_dataset/dataset.json"
TOOLMETA = f"{BIG}/data/gta_dataset/toolmeta.json"

CFG = f"{TACO}/configs"
FT = f"{TACO}/feature_tables"
RAW = f"{TACO}/raw_runs"
AUDIT = f"{TACO}/phase0_audit"
INTERV = f"{TACO}/interventions"
MODELS = f"{TACO}/models"
SEARCH = f"{TACO}/searches"
FROZEN = f"{TACO}/frozen_environments"
HELDOUT = f"{TACO}/heldout"
PLOTS = f"{TACO}/plots"
REPORTS = f"{TACO}/reports"
LOGS = f"{TACO}/logs"

SUBDIRS = [CFG, FT, RAW, AUDIT, INTERV, MODELS, SEARCH, FROZEN, HELDOUT,
           PLOTS, REPORTS, LOGS, f"{TACO}/paired_effects"]


def ensure():
    for d in SUBDIRS:
        os.makedirs(d, exist_ok=True)
    return TACO
