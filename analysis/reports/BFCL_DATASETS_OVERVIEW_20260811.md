# BFCL v1 datasets used in this investigation — statistics + raw samples

Date: 2026-08-11. Branch: `bfcl-token-likelihood-pilot`. Source for both:
[`gorilla-llm/Berkeley-Function-Calling-Leaderboard`](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
(Hugging Face). Both are BFCL v1 categories — the HF repo has renamed the
files in place to `BFCL_v3_*`, but `multiple`/`live_multiple` are the same
v1 category semantics (single-turn, non-executable, AST-scored).

Full raw files (unfiltered, exactly as downloaded):
- Dataset 1: `runtime/bfcl_v1_pilot_data/{questions,answers}.json`
- Dataset 2: `runtime/bfcl_live_multiple_data/{questions,answers}.json`

Open them directly with `python3 -c "import json; [print(json.loads(l)) for l in open('...')]"`
or any JSONL viewer — one JSON object per line, no wrapping array.

---

## Dataset 1 — `multiple` (`BFCL_v3_multiple.json`)

**What it is**: hand-authored (not real user traffic). Every task is
deliberately built with 2–4 candidate functions that plausibly compete for
the same query, exactly one of them correct — designed specifically to test
"can the model pick the right one out of a few similar options." Used for
the **forced-scoring pilot** (`BFCL_V1_TOKEN_LIKELIHOOD_PILOT_20260811.md`).

### Statistics

| Stat | Value |
|---|---|
| Tasks | 200 |
| Candidates/task | min 2, max 4, mean **2.79**, median 3 |
| Candidate-count histogram | 2 candidates: 79 tasks · 3: 85 · 4: 36 |
| GT functions/task | **all 200 have exactly 1** |

### Raw samples (first 3 tasks, verbatim, untruncated ground truth)

```json
{"id": "multiple_0", "question": [[{"role": "user", "content": "Can I find the dimensions and properties of a triangle, if I know its three sides are 5 units, 4 units and 3 units long?"}]], "function": [{"name": "triangle_properties.get", "description": "Retrieve the dimensions, such as area and perimeter, of a triangle if lengths of three sides are given.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "The length of first side of the triangle."}, "side2": {"type": "integer", "description": "The length of second side of the triangle."}, "side3": {"type": "integer", "description": "The length of third side of the triangle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of triangle. Default is true.", "default": true, "optional": true}, "get_perimeter": {"type": "boolean", "description": "A flag to determine whether to calculate the perimeter of triangle. Default is true.", "default": true, "optional": true}, "get_angles": {"type": "boolean", "description": "A flag to determine whether to calculate the internal angles of triangle. Default is true.", "default": true, "optional": true}}, "required": ["side1", "side2", "side3"]}}, {"name": "circle_properties.get", "description": "Retrieve the dimensions, such as area and circumference, of a circle if radius is given.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The length of radius of the circle."}, "get_area": {"type": "boolean", "description": "A flag to determine whether to calculate the area of circle. Default is true.", "default": true, "optional": true}, "get_circumference": {"type": "boolean", "description": "A flag to determine whether to calculate the circumference of circle. Default is true.", "default": true, "optional": true}}, "required": ["radius"]}}]}
GT: [{"triangle_properties.get": {"side1": [5], "side2": [4], "side3": [3], "get_area": ["", true], "get_perimeter": ["", true], "get_angles": ["", true]}}]
```

```json
{"id": "multiple_1", "question": [[{"role": "user", "content": "Calculate the area of a triangle, given the lengths of its three sides: 3, 4, and 5."}]], "function": [{"name": "math.triangle_area_heron", "description": "Calculates the area of a triangle using Heron's formula, given the lengths of its three sides.", "parameters": {"type": "dict", "properties": {"side1": {"type": "integer", "description": "Length of the first side of the triangle."}, "side2": {"type": "integer", "description": "Length of the second side of the triangle."}, "side3": {"type": "integer", "description": "Length of the third side of the triangle."}}, "required": ["side1", "side2", "side3"]}}, {"name": "math.circle_area", "description": "Calculates the area of a circle given its radius.", "parameters": {"type": "dict", "properties": {"radius": {"type": "float", "description": "The radius of the circle."}}, "required": ["radius"]}}, {"name": "math.triangle_area_base_height", "description": "Calculates the area of a triangle using the formula (1/2)base*height.", "parameters": {"type": "dict", "properties": {"base": {"type": "float", "description": "The base length of the triangle."}, "height": {"type": "float", "description": "The height of the triangle."}}, "required": ["base", "height"]}}]}
GT: [{"math.triangle_area_heron": {"side1": [3], "side2": [4], "side3": [5]}}]
```

```json
{"id": "multiple_2", "question": [[{"role": "user", "content": "What is the capital of Brazil?"}]], "function": [{"name": "country_info.largest_city", "description": "Fetch the largest city of a specified country.", "parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, "required": ["country"]}}, {"name": "country_info.capital", "description": "Fetch the capital city of a specified country.", "parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, "required": ["country"]}}, {"name": "country_info.population", "description": "Fetch the current population of a specified country.", "parameters": {"type": "dict", "properties": {"country": {"type": "string", "description": "Name of the country."}}, "required": ["country"]}}]}
GT: [{"country_info.capital": {"country": ["Brazil"]}}]
```

Note the pattern in all three: candidates are near-synonyms of the same
family (`triangle_properties.get` vs `circle_properties.get`;
`math.triangle_area_heron` vs `math.circle_area` vs
`math.triangle_area_base_height`; `country_info.capital` vs `.largest_city`
vs `.population`) — this is deliberately constructed confusability, not
organic reuse.

---

## Dataset 2 — `live_multiple` (`BFCL_v3_live_multiple.json`)

**What it is**: sourced from **real user traffic** submitted to the BFCL
leaderboard ("live" = pulled from actual logged calls, not hand-written).
Candidate-list size varies a lot (2–37) because it reflects however many
functions a real integration actually registered, not a designed-for-testing
count. Used for the **pool-subset experiment**
(`BFCL_POOL_SUBSET_20260811.md`) — tasks are grouped into shared pools to
build a genuinely cluttered candidate space (see that report for why).

### Statistics

| Stat | Value |
|---|---|
| Tasks (raw file) | 1053 |
| Tasks with a `possible_answer` entry | 1052 (`live_multiple_1052-79-0` has none, skipped everywhere) |
| Candidates/task (own list, before any pooling) | min 2, max **37**, mean **3.97**, median 4 |
| Candidate-count buckets | 2–4: 702 tasks · 5–9: 335 · 10–19: 14 · 20+: 2 |
| GT functions/task | **all 1052 have exactly 1** |
| Distinct function names, whole dataset | **457** |
| Total (task, function) pairs | 4178 |
| Reuse factor (pairs ÷ distinct names) | **9.14** — the same function shows up in ~9 different tasks' candidate lists on average |

**Derived: round-robin batching used by the pool-subset experiment**
(`task_idx % 150` → 150 batches, each batch's tasks share the union of their
candidate functions as one pool):

| Stat | Value |
|---|---|
| Batches | 150 |
| Tasks/batch | min 7, max 8, mean 7.01 |
| Pool size/batch (deduped) | min **15**, max **57**, mean **24.37**, median 25 |

### Raw samples (verbatim, chosen to span the candidate-count range)

**Small end (2 candidates)** — `live_multiple_0-0-0`:
```json
{"id": "live_multiple_0-0-0", "question": [[{"role": "user", "content": "update my latte to a large size with coconut milk and make it extra sweet? make it served 'boiling hot' as the special request. The drink id is 'latte'"}]], "function": [{"name": "ChaFod", "description": "Changes the food item based on the customer's request, allowing for modifications to the ingredients or preparation method.", "parameters": {"type": "dict", "required": ["foodItem"], "properties": {"foodItem": {"type": "string", "description": "The name of the food item to be modified as requested by the customer."}, "newIngredients": {"type": "string", "description": "A comma-separated list of new ingredients to include in the food item, if any.", "default": ""}, "removeIngredients": {"type": "string", "description": "A comma-separated list of ingredients to remove from the food item, if any.", "default": ""}, "specialInstructions": {"type": "string", "description": "Special preparation instructions..."}}}}, {"name": "ChaDri.change_drink", "description": "..."}]}
GT: [{"ChaDri.change_drink": {"drink_id": ["latte"], "new_preferences": [{"size": ["large"], "milk_type": ["coconut"], "sweetness_level": ["extra"], "temperature": ["hot"], "special_instructions": ["served boiling hot", "serve boiling hot", "boiling hot", "served 'boiling hot'"]}]}}]
```
Note `special_instructions` accepts **multiple phrasings as equally
correct** — BFCL's AST scoring allows a list of acceptable values per
argument, not just one exact string. (Our matching, by contrast, only checks
the *function name* was chosen correctly, not argument correctness — see
Caveats in both experiment reports.)

**Mid-range (4 candidates)** — `live_multiple_5-3-0`:
```json
{"id": "live_multiple_5-3-0", "question": [[{"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."}, {"role": "user", "content": "Could you tell me the current weather conditions in Shanghai, using the metric system?"}]], "function": [{"name": "get_current_weather", "description": "Retrieves the current weather conditions for a specified location, such as 'City, State' or 'City, Country'.", "parameters": {"type": "dict", "required": ["location"], "properties": {"location": {"type": "string", "description": "..."}, "unit": {"type": "string", "description": "...", "enum": ["metric", "imperial"], "default": "metric"}}}}, {"name": "start_oncall", "description": "Initiate an on-call..."}, "... 2 more"]}
GT: [{"get_current_weather": {"location": ["Shanghai, China"], "unit": ["", "metric"]}}]
```
Note this one carries a **system prompt** (`"Don't make assumptions..."`) —
some `live_multiple` tasks have a system turn, most don't; our prompt
builder only reads the `user` turn's content, so system-turn instructions
like this one are currently **dropped**, not passed to the model (a gap
worth flagging — see below).

**Large end (37 candidates, the dataset max)** — `live_multiple_985-216-0`:
```json
{"id": "live_multiple_985-216-0", "question": [[{"role": "user", "content": "User query: I need to mark my reminders as completed using my authentication token '1231289312'.\nPlan step 1: Use the authentication token to mark the reminders as completed.\nAPI response: "}]], "function": [{"name": "reminders_complete", "description": "Marks specified reminders as completed and returns the status of the operation.", "parameters": {"type": "dict", "required": ["token"], "properties": {"token": {"type": "string", "description": "Authentication token to verify the user's identity."}}}}, {"name": "reminders_delete", "description": "..."}, {"name": "reminders_info", "description": "..."}, "... 34 more"]}
GT: [{"reminders_complete": {"token": ["1231289312"]}}]
```
Note the query itself is an unusual format — `"User query: ... \nPlan step
1: ... \nAPI response: "` — this looks like it was captured mid-agent-trace
from a real multi-step tool-use system, not a plain user question. This
task's own 37-candidate menu already exceeds this experiment's built pools'
max (57 is bigger, but only because pooling *combines* several tasks —
this is the largest single task's own native list).

---

## Two things worth flagging from actually reading the raw data (not just the aggregates)

1. **BFCL's own ground truth allows multiple acceptable argument
   phrasings** (see the `special_instructions` example above) — our
   experiments only score whether the **function name** matches, never
   whether arguments are correct. This means our accuracy numbers are
   measuring a strictly easier sub-problem than BFCL's own official
   scoring (which is AST-based and checks arguments too).
2. **Some `live_multiple` tasks carry a system-prompt turn** that our
   prompt builder (`build_menu_prefix` / `build_prefix`) currently drops —
   only `role: user` content is read. This is a real, previously
   undocumented gap: for tasks like `live_multiple_5-3-0`, the model is
   missing context ("don't guess ambiguous values") that BFCL's own
   evaluation presumably includes. Quantified: **37/1053 (3.5%)** of
   `live_multiple` tasks have a system turn (`multiple` has none, 0/200) —
   a small enough share that it's unlikely to explain the headline
   full-vs-topk gap, but it means ~36 of the pool-subset experiment's 1052
   scored tasks were run with less context than BFCL's own protocol
   presumably intends. Not corrected for in either report to date.

## Files

- Dataset 1: `runtime/bfcl_v1_pilot_data/{questions,answers}.json` (200 lines each, JSONL)
- Dataset 2: `runtime/bfcl_live_multiple_data/{questions,answers}.json` (1053 / 1053 lines, JSONL)
- Fetch scripts: `experiments/bfcl_token_likelihood/fetch_data.sh` (dataset 1); dataset 2 was fetched ad hoc with the same `curl` pattern against `BFCL_v3_live_multiple.json` — not yet scripted, worth adding if this dataset gets reused.
