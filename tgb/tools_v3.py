"""TGB-v3 tools: upstream sources that are not OCR.

v1 and v2 are extraction benchmarks in disguise -- OCR is in 3/3 and 8/8 of
their chains respectively, which is the same criticism the paper levels at
GTA-Atomic. Every likelihood result so far is therefore conditional on a
read-text-then-compute pipeline, and the answer-echo mechanism that §4a blames
for the real-GTA null lives exactly there.

v3 keeps one OCR family as an inside-the-benchmark control and sources the rest
from elsewhere:

    counting          CountGivenObject on an image with no text at all
    attributes        RegionAttributeDescription on rendered shapes
    retrieval         GoogleSearch over a nonce corpus, no image
    structured data   TableQuery, a new deterministic record lookup
    instruments       SensorAPI, a new deterministic reading service
    nothing           the numbers are already in the question

The last one matters twice over: it gives the benchmark its first tool-free and
tool-optional tasks, without which dA collapses onto Acc and the correlation
the meeting plan asks for is degenerate.

Same three-state contract as v2: in-chain a tool consumes its upstream's parsed
value; fired blind it answers from whatever it can see and is therefore
misleading; orphaned it errors.
"""
import re

from .tools import DEFAULT_ARGS, TOOLS
from .tools import count_given_object as _count_v1
from .tools_v2 import _resolve

_NUM = re.compile(r"(\d+(?:\.\d+)?)")


def table_query(scene, boxes, rng, corrupt=False, key=None, **kw):
    """Look a record up in the task's own table. Needed by `table_lookup`;
    fired blind (no key) it returns the first record, which is a plausible but
    wrong row."""
    rows = scene.get("table")
    if not rows:
        return "TableQuery: no table is attached to this request."
    hit = None
    if key:
        hit = next((r for r in rows if str(r["id"]).lower() in str(key).lower()),
                   None)
    if hit is None:
        hit = rows[0]
    if corrupt:
        alts = [r for r in rows if r is not hit]
        if alts:
            hit = alts[rng.randrange(len(alts))]
    return (f"TableQuery: record {hit['id']} | {hit['label']} | "
            f"quantity {hit['qty']} | unit value {hit['unit']:.2f}")


def sensor_api(scene, boxes, rng, corrupt=False, station=None, **kw):
    """Read an instrument. Needed by `sensor_convert`; blind it returns the
    first station's reading."""
    st = scene.get("stations")
    if not st:
        return "SensorAPI: no station in range."
    hit = None
    if station:
        hit = next((s for s in st if str(s["id"]).lower() in str(station).lower()),
                   None)
    if hit is None:
        hit = st[0]
    v = hit["reading"]
    if corrupt:
        v = round(v * rng.choice([0.86, 1.15]), 2)
    return f"SensorAPI: station {hit['id']} reports {v:g} {hit['unit']}"


def region_attribute(scene, boxes, rng, corrupt=False, value=None, **kw):
    """Report the measured attribute of the region the question asks about.

    This is the non-OCR perception channel: the answer depends on *seeing* a
    rendered size, not on reading any glyph. Blind, it reports the first
    shape's attribute rather than the asked-for one.
    """
    sh = scene.get("shapes")
    if not sh:
        return ("The region contains no measurable object.")
    target = None
    if value:
        target = next((s for s in sh if s["name"] in str(value)), None)
    if target is None:
        target = sh[0]
    v = target["size"]
    if corrupt:
        v = max(1, v + rng.choice([-9, 11]))
    return (f"The {target['color']} {target['name']} occupies a region of "
            f"roughly {v} units across; the remaining objects in the frame are "
            f"smaller and of assorted colours.")


def count_given_object(scene, boxes, rng, corrupt=False, text=None, **kw):
    """Counting on the textless `objects` scene, delegating anything else back
    to the v1 executor so already-generated v1/v2 data stays byte-identical."""
    if scene.get("kind") != "objects":
        return _count_v1(scene, boxes, rng, corrupt=corrupt, text=text)
    n = scene["n_target"]
    if text and scene["target"] not in text.lower():
        for d in scene["distractors"]:
            if d["name"] in text.lower():
                n = d["count"]
                break
    if corrupt:
        n = max(1, n + rng.choice([-2, -1, 1, 2]))
    return str(n)


V3_TOOLS = {
    "CountGivenObject": count_given_object,
    "TableQuery": table_query,
    "SensorAPI": sensor_api,
    "RegionAttributeDescription": region_attribute,
}


def register():
    TOOLS.update(V3_TOOLS)
    DEFAULT_ARGS.update({"TableQuery": {}, "SensorAPI": {}})
