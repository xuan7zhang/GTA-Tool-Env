# Clustering BFCL v1 `live_multiple` tasks by tool-space similarity

Date: 2026-08-11. Branch: `bfcl-token-likelihood-pilot`. No model inference
involved — this is a purely offline, structural analysis of the dataset
(candidate-tool sets), not a likelihood or accuracy experiment.

## Question

The forced-scoring and pool-subset experiments both rank/prune a task's
candidate pool **online, per task**, at inference time — every task pays for
its own scoring pass. This raises a natural follow-up: **do many tasks
already share close to the same tool subset**, such that most of that online
cost could be replaced by an offline lookup ("which domain is this task in →
here's its cached candidate subset"), falling back to online scoring only
for tasks that don't fit a known domain?

Clustering tasks by tool-space similarity answers that structural question
directly, with no model calls needed.

## Method

Each of the 1052 `live_multiple` tasks is represented as a **multi-hot
binary vector** over the 457 distinct function names in the dataset (1 = one
of this task's candidate tools, 0 = not). Rows are **L2-normalized** so
ordinary (Euclidean) k-means behaves like spherical/cosine k-means —
appropriate here because what matters is *which tools overlap* between two
tasks, not raw distance in a sparse 457-dimensional {0,1} space.

**Choosing K.** Silhouette score was swept over K ∈ {10, 20, 30, 40, 50, 60,
80}:

| K | Silhouette | Inertia |
|---:|---:|---:|
| 10 | 0.266 | 611.7 |
| 20 | 0.353 | 467.4 |
| 30 | 0.426 | 370.5 |
| **40** | **0.494** | 306.7 |
| 50 | 0.549 | 252.8 |
| 60 | 0.625 | 209.6 |
| 80 | 0.736 | 151.9 |

Silhouette rises monotonically with K rather than showing a classic elbow —
expected, since the underlying data already has 233 *exactly*-identical
candidate-sets among the 1052 tasks (some shared by 20-38 tasks, 107 shared
by only 1), so allowing more clusters keeps letting k-means carve out
another exact-duplicate group with near-zero within-cluster distance. There
is no single "correct" K here; **K=40** was picked for being coarse enough
to read by hand (40 clusters, not 233) while still separating clearly
distinct domains (see below) rather than lumping everything together (K=10
mixed unrelated domains — inspected but not included here).

Implementation: `experiments/bfcl_token_likelihood/cluster_tasks_by_tool_space.py`.

## Results: 40 clusters, most read as real business domains

Sorted by size, with each cluster's top-5 most frequent candidate tools and
a coherence check — **`n_distinct_gt_tools / n_tasks`**: a low ratio means
most tasks in the cluster share a small set of ground-truth answers (the
cluster is a tight, reusable domain); a high ratio near 1 means almost every
task has a different correct answer (the cluster is a grab-bag, not a real
domain).

| Cluster | n tasks | % of dataset | distinct GT tools | ratio | Coherence | Top tools |
|---:|---:|---:|---:|---:|---|---|
| 26 | 122 | 11.6% | 97 | 0.80 | **incoherent** | `multiply`, `add`, `sub` |
| 8 | 62 | 5.9% | 3 | 0.05 | coherent | `Media_3_FindMovies`, `Media_3_PlayMovie`, `Music_3_PlayMedia` — media/music playback |
| 2 | 60 | 5.7% | 4 | 0.07 | coherent | `Movies_1_BuyMovieTickets`, `Movies_1_FindMovies`, `Movies_1_GetTimesForMovie` — movie tickets |
| 7 | 57 | 5.4% | 3 | 0.05 | coherent | `Events_3_FindEvents`, `Events_3_BuyEventTickets`, `Movies_3_FindMovies` — events |
| 9 | 48 | 4.6% | 4 | 0.08 | coherent | `Restaurants_2_ReserveRestaurant`, `Restaurants_2_FindRestaurants` — dining |
| 30 | 47 | 4.5% | 4 | 0.09 | coherent | `Payment_1_RequestPayment`, `Payment_1_MakePayment`, `Restaurants_2_ReserveRestaurant` — payments |
| 13 | 45 | 4.3% | 3 | 0.07 | coherent | `Homes_2_FindHomeByArea`, `Homes_2_ScheduleVisit`, `RentalCars_3_GetCarsAvailable` — home search |
| 25 | 38 | 3.6% | 3 | 0.08 | coherent | `Music_3_PlayMedia`, `Music_3_LookupMusic`, `Weather_1_GetWeather`, `RideSharing_2_GetRide` — rideshare/misc |
| 14 | 38 | 3.6% | 2 | 0.05 | coherent | `get_service_providers`, `view_service_provider_profile` — service marketplace |
| 29 | 36 | 3.4% | 4 | 0.11 | coherent | `Hotels_4_ReserveHotel`, `Hotels_4_SearchHotel`, `Travel_1_FindAttractions` — hotel/travel |
| 18 | 33 | 3.1% | 12 | 0.36 | mixed | `get_current_weather`, `add_postgres_server`, `dartfx_help` — dev-tools grab-bag |
| 5 | 33 | 3.1% | 8 | 0.24 | mixed | `add_postgres_server`, `add_mtnards_server`, `dartfx_help` — server config |
| 0 | 32 | 3.0% | 3 | 0.09 | coherent | `Flights_4_SearchOnewayFlight`, `RentalCars_3_GetCarsAvailable` — flights/cars |
| 6 | 29 | 2.8% | 4 | 0.14 | coherent | `Alarm_1_GetAlarms`, `Alarm_1_AddAlarm`, `Services_1_BookAppointment` — alarms/appointments |
| 27 | 27 | 2.6% | 3 | 0.11 | coherent | `Buses_3_FindBus`, `Buses_3_BuyBusTicket`, `Events_3_FindEvents` — buses/events |
| 16 | 26 | 2.5% | 3 | 0.12 | coherent | `Flights_4_SearchOnewayFlight`, `Hotels_2_BookHouse` — flights/hotels |
| 32 | 25 | 2.4% | 2 | 0.08 | coherent | `Hotels_2_BookHouse`, `Hotels_2_SearchHouse`, `Travel_1_FindAttractions` — home rentals |
| 10 | 21 | 2.0% | 2 | 0.10 | coherent | `Services_4_BookAppointment`, `Services_4_FindProvider`, `Weather_1_GetWeather` |
| 1 | 21 | 2.0% | 1 | 0.05 | coherent | `Movies_3_FindMovies`, `Music_3_PlayMedia`, `Music_3_LookupMusic` |
| 23 | 17 | 1.6% | 5 | 0.29 | mixed | `OpenWeatherMap.get_current_weather`, `ControlAppliance.execute`, `HNA_NEWS.search` — smart-home grab-bag |
| 4 | 17 | 1.6% | 1 | 0.06 | coherent | `Buses_3_FindBus`, `Buses_3_BuyBusTicket`, `Events_3_FindEvents` |
| 33 | 16 | 1.5% | 8 | 0.50 | mixed | `uber.ride`, `uber.eat.order`, `flight.status.check` |
| 11 | 16 | 1.5% | 5 | 0.31 | mixed | `set_alarm`, `set_volume`, `play_song` |
| 15 | 15 | 1.4% | 4 | 0.27 | mixed | `get_sensor_readings_latest`, `get_sensor_readings_history_by_interval` — IoT sensors |
| 22 | 15 | 1.4% | 5 | 0.33 | mixed | `get_detail_adriel_project`, `get_adriel_list_projects`, `get_adriel_profile` — portfolio API |
| 21 | 15 | 1.4% | 2 | 0.13 | coherent | `version_api.VersionApi.get_version`, `project_api...` — dev/package-registry API |
| 20 | 15 | 1.4% | 1 | 0.07 | coherent | `Media_3_FindMovies`, `Media_3_PlayMovie`, `Weather_1_GetWeather` |
| 19 | 14 | 1.3% | 4 | 0.29 | mixed | `inventory_management`, `product_search`, `order_status_check` — e-commerce |
| 12 | 13 | 1.2% | 6 | 0.46 | mixed | `detail_project`, `detail_experience_and_education`, `list_projects` — portfolio API |
| 31 | 13 | 1.2% | 1 | 0.08 | coherent | `Events_3_FindEvents`, `Events_3_BuyEventTickets`, `Hotels_4_ReserveHotel` |
| 3 | 12 | 1.1% | 2 | 0.17 | mixed | `Trains_1_GetTrainTickets`, `Trains_1_FindTrains`, `Hotels_2_BookHouse` |
| 34 | 11 | 1.0% | 4 | 0.36 | mixed | `analysis_api...`, `acl_api...` — internal platform API |
| 17 | 11 | 1.0% | 3 | 0.27 | mixed | `Trains_1_GetTrainTickets`, `Trains_1_FindTrains`, `Payment_1_RequestPayment` |
| 36 | 10 | 1.0% | 4 | 0.40 | mixed | `stock_price.get`, `weather.get`, `weather.get_weather` |
| 28 | 10 | 1.0% | 1 | 0.10 | coherent | `RideSharing_2_GetRide`, `Travel_1_FindAttractions`, `Weather_1_GetWeather` |
| 35 | 9 | 0.9% | 3 | 0.33 | mixed | `CustomDashboardsApi...`, `api_token_api...` — dashboards platform API |
| 37 | 8 | 0.8% | 5 | 0.62 | **incoherent** | `detail_adriel_project`, `adriel_detail_experience_and_education` |
| 38 | 7 | 0.7% | 4 | 0.57 | **incoherent** | `search_engine.query`, `generate_image`, `generate_human_image` |
| 24 | 5 | 0.5% | 2 | 0.40 | mixed | `generate_image_tool`, `tts_tool`, `write_markdown_tool` |
| 39 | 3 | 0.3% | 2 | 0.67 | **incoherent** | `get_pods`, `get_services` — Kubernetes-ish, too few tasks to tell |

**Roll-up by coherence** (threshold: ratio ≤ 0.15 = coherent, ≤ 0.5 = mixed, > 0.5 = incoherent):

| Coherence | Clusters | Tasks | Share of dataset |
|---|---:|---:|---:|
| Coherent | 21 | 682 | 64.8% |
| Mixed | 15 | 230 | 21.9% |
| Incoherent | 4 | 140 | 13.3% |

**Almost two-thirds of the dataset (682/1052 tasks) falls into 21 tightly
coherent domain clusters** — real, recognizable business domains (movie
tickets, restaurant reservations, home search, hotel booking, alarms,
service marketplaces, etc.) where a handful of tasks' worth of candidate
tools cover the whole cluster.

## The one cluster that doesn't work: cluster 26

122 tasks (11.6% of the whole dataset) landed in one cluster with 97
distinct ground-truth answers — essentially no two tasks share an answer.
This is the classic k-means failure mode on sparse high-dimensional binary
data: **tasks whose tool set barely overlaps with anything else get pulled
into whichever cluster's centroid happens to be nearest**, producing an
incoherent "leftover bin" rather than a real group. This lines up with the
dataset's own structure — 107 of the 233 exact-candidate-set groups in the
full dataset occur in only one task each, i.e. roughly 10% of tasks really
are one-off, not clusterable by tool overlap at all.

## Fix: DBSCAN instead of forcing every task into a k-means bucket

Cluster 26 is a k-means-specific artifact, not a fact about the data: k-means
must assign **every** point to one of K clusters, so a task whose tool set
barely overlaps with anything else still gets pulled to whichever centroid
happens to be nearest — even when that fit is bad. Checked directly: of
cluster 26's 122 tasks, only 9 actually contain one of the cluster's own
"top 5 tools" (the `multiply`/`add`/`fahrenheit_to_celsius` math-utility
group, which is really its own small 7-9-task cluster, not 122 tasks' worth
of signal); 19/122 have candidate tools that appear in *no other task in the
dataset at all*. The "top tools" label was never representative — it's just
the most frequent tools among an otherwise unrelated leftover group.

**DBSCAN** (density-based clustering) fixes this directly: it explicitly
labels points that don't belong to any sufficiently dense region as
**noise** (cluster `-1`) instead of forcing them into the nearest cluster.
Same feature vectors, cosine distance (`cosine_distances` on the same
L2-normalized multi-hot vectors), `eps` swept:

| eps | Clusters | Noise | Noise % |
|---:|---:|---:|---:|
| 0.15 | 75 | 191 | 18.2% |
| 0.20 | 63 | 171 | 16.3% |
| 0.25 | 58 | 156 | 14.8% |
| **0.30** | **34** | **148** | **14.1%** |
| 0.35 | 33 | 143 | 13.6% |
| 0.40 | 24 | 139 | 13.2% |

`eps=0.3, min_samples=4` picked for a cluster count comparable to the
k-means run (34 vs. 40) — not separately tuned beyond this sweep.

**Before / after** (same coherence metric — `n_distinct_gt_tools / n_tasks`
per cluster — applied to both):

| | k-means (K=40) | DBSCAN (eps=0.3) |
|---|---:|---:|
| Coherent clusters / tasks | 21 / 682 | 12 / 677 |
| Mixed clusters / tasks | 15 / 230 | 17 / 195 |
| **Incoherent clusters / tasks** | **4 / 140** | **5 / 32** |
| Explicit "no good cluster" (noise) | 0 (forced into incoherent bins) | 148 (14.1%, honestly labeled) |

The coherent-task count barely moved (682 → 677) — DBSCAN isn't
"discovering" fewer good clusters, it's **correctly refusing to manufacture
fake ones**: the 140 incoherent k-means tasks mostly weren't wrong so much
as unclusterable, and DBSCAN now says so directly (148 noise) instead of
hiding them inside a misleadingly-labeled cluster 26. Incoherent-task count
dropped 140 → 32, a real improvement, not just relabeling — DBSCAN also
merged some domains k-means had artificially split apart: its largest
cluster is 268 tasks (likely folding several of k-means's separate
media/music-adjacent clusters — 8, 25, 1, 20 — into one), well above
k-means's largest *legitimate* cluster of 62 (cluster 8). K-means is
constrained to roughly-equal-sized Voronoi-like cells by construction
(forcing exactly K partitions of comparable pull), which both artificially
splits large natural domains and creates the leftover-bin problem for
small ones; DBSCAN has neither constraint.

**Practical implication for the two-tier design below**: the noise bucket
*is* the honest answer to "which tasks can't be served by an offline
domain-cache lookup" — 14.1% needing the online fallback path, not the 13.3%
k-means implied *plus* however many of its "coherent"-looking clusters were
secretly less clean than they looked.

**Full DBSCAN cluster table** (34 clusters + noise, same coherence metric):

| Cluster | n tasks | % of dataset | distinct GT tools | ratio | Coherence | Top tools |
|---:|---:|---:|---:|---:|---|---|
| 19 | 268 | 25.5% | 14 | 0.05 | coherent | `Events_3_FindEvents`, `Events_3_BuyEventTickets`, `Restaurants_2_ReserveRestaurant` |
| 22 | 121 | 11.5% | 5 | 0.04 | coherent | `Music_3_PlayMedia`, `Music_3_LookupMusic`, `Media_3_FindMovies` |
| 20 | 75 | 7.1% | 7 | 0.09 | coherent | `Hotels_2_BookHouse`, `Hotels_2_SearchHouse`, `Travel_1_FindAttractions` |
| 24 | 60 | 5.7% | 4 | 0.07 | coherent | `Movies_1_BuyMovieTickets`, `Movies_1_FindMovies`, `Movies_1_GetTimesForMovie` |
| 4 | 48 | 4.6% | 8 | 0.17 | mixed | `add_postgres_server`, `dartfx_help`, `add_mtnards_server` |
| 25 | 38 | 3.6% | 4 | 0.11 | coherent | `Homes_2_FindHomeByArea`, `Homes_2_ScheduleVisit`, `Alarm_1_GetAlarms` |
| 30 | 38 | 3.6% | 2 | 0.05 | coherent | `get_service_providers`, `view_service_provider_profile` |
| 18 | 20 | 1.9% | 2 | 0.10 | coherent | `Flights_4_SearchOnewayFlight`, `Flights_4_SearchRoundtripFlights`, `RentalCars_3_GetCarsAvailable` |
| 31 | 16 | 1.5% | 5 | 0.31 | mixed | `set_alarm`, `set_volume`, `play_song` |
| 0 | 15 | 1.4% | 5 | 0.33 | mixed | `OpenWeatherMap.get_current_weather`, `ControlAppliance.execute`, `HNA_WQA.search` |
| 8 | 15 | 1.4% | 4 | 0.27 | mixed | `get_sensor_readings_latest`, `get_sensor_readings_history_by_interval`, `get_sensor_readings_history` |
| 26 | 15 | 1.4% | 1 | 0.07 | coherent | `Media_3_FindMovies`, `Media_3_PlayMovie`, `Weather_1_GetWeather` |
| 29 | 15 | 1.4% | 1 | 0.07 | coherent | `Homes_2_FindHomeByArea`, `Homes_2_ScheduleVisit`, `RentalCars_3_GetCarsAvailable` |
| 2 | 14 | 1.3% | 4 | 0.29 | mixed | `inventory_management`, `product_search`, `order_status_check` |
| 12 | 13 | 1.2% | 6 | 0.46 | mixed | `detail_project`, `detail_experience_and_education`, `list_projects` |
| 23 | 10 | 1.0% | 1 | 0.10 | coherent | `Alarm_1_GetAlarms`, `Alarm_1_AddAlarm`, `Messaging_1_ShareLocation` |
| 27 | 10 | 1.0% | 1 | 0.10 | coherent | `RideSharing_2_GetRide`, `Travel_1_FindAttractions`, `Weather_1_GetWeather` |
| 10 | 9 | 0.9% | 4 | 0.44 | mixed | `get_detail_adriel_project`, `get_adriel_list_projects`, `get_adriel_experiences_and_education` |
| 13 | 9 | 0.9% | 2 | 0.22 | mixed | `project_api...get_project_by_name_and_version`, `version_api...get_version`, `badge_api...` |
| 21 | 9 | 0.9% | 2 | 0.22 | mixed | `Flights_4_SearchOnewayFlight`, `Flights_4_SearchRoundtripFlights`, `Hotels_4_ReserveHotel` |
| 5 | 8 | 0.8% | 3 | 0.38 | mixed | `acl_api...retrieve_projects`, `analysis_api...retrieve_analysis`, `acl_api.add_mapping` |
| 11 | 8 | 0.8% | 5 | 0.62 | **incoherent** | `detail_adriel_project`, `adriel_detail_experience_and_education`, `adriel_list_projects` |
| 1 | 7 | 0.7% | 4 | 0.57 | **incoherent** | `multiply`, `add`, `sub` |
| 6 | 7 | 0.7% | 4 | 0.57 | **incoherent** | `search_engine.query`, `generate_image`, `generate_human_image` |
| 28 | 7 | 0.7% | 1 | 0.14 | coherent | `Payment_1_RequestPayment`, `Payment_1_MakePayment`, `Trains_1_GetTrainTickets` |
| 33 | 7 | 0.7% | 2 | 0.29 | mixed | `api_token_api...get_api_tokens`, `CustomDashboardsApi...`, `api_token_api...post_api_token` |
| 7 | 6 | 0.6% | 2 | 0.33 | mixed | `weather.get`, `stock_price.get` |
| 9 | 6 | 0.6% | 3 | 0.50 | mixed | `get_detail_adriel_project`, `get_adriel_experiences`, `get_adriel_education` |
| 17 | 6 | 0.6% | 1 | 0.17 | mixed | `version_api...get_version`, `project_api...update_project`, `oidc_api...is_available` |
| 3 | 5 | 0.5% | 3 | 0.60 | **incoherent** | `uber.ride`, `uber.eat.order`, `get_current_weather` |
| 14 | 5 | 0.5% | 1 | 0.20 | mixed | `get_current_weather`, `get_area_of_square`, `get_contact_information` |
| 15 | 5 | 0.5% | 3 | 0.60 | **incoherent** | `set_integer`, `set_string`, `set_float` |
| 16 | 5 | 0.5% | 2 | 0.40 | mixed | `generate_image_tool`, `tts_tool`, `write_markdown_tool` |
| 32 | 4 | 0.4% | 2 | 0.50 | mixed | `uber.eat.order`, `flight.status.check` |
| **noise** | **148** | **14.1%** | 112 | — | **noise** | `get_current_weather`, `uber.ride`, `Messaging_1_ShareLocation` |

Notable difference from the k-means table: DBSCAN merged several of
k-means's separate media/movies/events/restaurants clusters (8, 2, 7, 9, ...)
into one 268-task cluster (19) because their tool sets overlap heavily —
k-means's equal-cell-size pull had been artificially splitting one real,
large domain into several smaller ones.

## Why this matters for tool-space optimization

The forced-scoring / pool-subset experiments both re-rank a task's full
candidate pool **online**, at inference time, for every task — that's the
validated, accurate approach, but it costs one scoring call per candidate
tool per task. This clustering result suggests a **two-tier design** worth
testing as a follow-up:

1. **Offline**: cluster the tool catalog's historical tasks (as done here)
   and cache a candidate subset per coherent cluster — for the ~65% of
   tasks that land in one of the 21 coherent domains, a cheap
   "which domain does this new task look like" classification (could reuse
   the same multi-hot-vector nearest-centroid check) plus a table lookup
   replaces the online scoring pass entirely.
2. **Online fallback**: for tasks that don't clearly match a coherent
   cluster (the ~13% incoherent share, plus low-confidence cluster
   assignments in the mixed tier), fall back to the validated per-task
   forced-scoring approach from the pool-subset experiment.

This was a structural observation only when first written — **it has since
been tested for accuracy, and the naive version of the design does not
work.** See the next section.

## Tested: the naive two-tier design underperforms the trivial baseline

Implementation: `experiments/bfcl_token_likelihood/cluster_cache_experiment.py`.
Real unconstrained generation (temperature 0), same model/endpoint as every
other experiment in this investigation. For every task in a DBSCAN
(non-noise) cluster, built a **leave-one-out cache**: the union of
ground-truth tool names from every *other* task in the same cluster
(excluding the task itself, so nothing leaks — a task's own answer can only
appear in its cache if some other cluster-mate also needed that tool).
Compared against simply showing the model the task's own BFCL-provided
candidate list (`native_full`) — no clustering, no scoring, the cheapest
possible baseline.

| Condition | n | Accuracy |
|---|---:|---:|
| `native_full` (all tasks, no optimization at all) | 1052 | **95.0%** |
| `native_full` (clustered-tasks subset only, for apples-to-apples) | 904 | 95.5% |
| `cluster_cache` (leave-one-out domain cache, clustered tasks) | 904 | **86.1%** |
| Blended two-tier system (`cluster_cache` + `native_full` fallback on noise) | 1052 | 86.9% |

**The cluster-cache condition loses to `native_full` by ~9 points, on the
same tasks.** Diagnosis, not just the headline number:

- **Recall isn't the problem.** The leave-one-out cache actually contained
  the ground-truth tool 97.6% of the time (882/904) — close to the pool-
  subset experiment's online-scoring recall (97.9%).
- **Even when the cache had the right tool, the model picked wrong more
  often than with `native_full`**: 88.2% accuracy conditional on
  `gt_in_cache=True`, vs. 95.5% for `native_full` on the same tasks. The
  cache's contents are the problem, not its coverage.
- **The cache is bigger, not smaller, than what it replaces**: mean cache
  size 7.14 tools vs. mean native-list size 4.03. There's no "fewer
  candidates" win here at all.
- **Likely mechanism**: a cluster's cache is "every ground-truth answer
  ever needed by this domain's tasks" — by construction, every tool in it
  is a *real, correct answer for some task in this domain*, so they're all
  domain-plausible and genuinely harder to tell apart than BFCL's own
  native candidate list, which was curated per-task to include
  distractors specific to that one question. Domain co-membership
  (`unstructured GT union`) turns out to be a much noisier relevance signal
  than either BFCL's own per-task curation or this investigation's online
  forced-scoring ranking.

**Takeaway**: clustering by tool-space similarity is a valid, useful
*diagnostic* (it correctly recovers real business domains, see the tables
above) — but "cache the union of a domain's historical correct answers" is
not a good way to build a cheap offline candidate-selection system. A
follow-up worth trying instead of abandoning the idea: apply the *already-
validated* online forced-scoring ranking to the cache's contents too (rank
the cluster's cached tools against the new task's query, keep only the
top-K) rather than showing the whole undifferentiated cache — this has not
been tried.

## Caveats

- **No accuracy claim here.** This is purely a description of the dataset's
  structure (which tasks share which tools), not an evaluation of any
  selection method.
- **K=40 was chosen for readability, not by a principled elbow** — silhouette
  score keeps rising with K because ~233 exact-duplicate candidate-sets
  exist in the data; a different K would redraw cluster boundaries
  (finer K would likely split some of the "mixed" clusters into cleaner
  coherent ones and shrink the incoherent bin further, at the cost of more
  clusters to read).
- **Coherence thresholds (0.15 / 0.5) are an ad hoc heuristic**, not a
  validated cutoff — used here only to make the roll-up table legible.
- Only run on `live_multiple` (the dataset with enough tool reuse across
  tasks for clustering to be meaningful); `multiple`'s 200 tasks are
  hand-designed to be mostly domain-distinct from each other and weren't
  clustered.
- **DBSCAN's `eps=0.3` / `min_samples=4` were picked from a 6-point sweep
  for a cluster count comparable to the k-means run, not independently
  tuned or validated** — the noise-percentage curve (18.2% → 13.2% across
  the swept range) shows the noise/cluster-count tradeoff is fairly
  continuous, not a sharp elbow either.
- Both methods share the same underlying representational choice (multi-hot
  vector over exact function names, L2-normalized, cosine-like distance) —
  neither accounts for *semantic* similarity between differently-named
  tools that do similar things, which a description-embedding-based
  distance (e.g. reusing this repo's `relevance_selector.py` embeddings)
  might cluster differently.

## Artifacts

- Clustering code: `experiments/bfcl_token_likelihood/cluster_tasks_by_tool_space.py`
  (`--method kmeans` or `--method dbscan`)
- K-means (K=40) cluster assignments: `runtime/bfcl_live_multiple_data/kmeans_k40_clusters.json`
- DBSCAN (eps=0.3) cluster assignments, including the noise bucket (cluster `-1`):
  `runtime/bfcl_live_multiple_data/dbscan_eps0.3_clusters.json`
- Cluster-cache accuracy test code: `experiments/bfcl_token_likelihood/cluster_cache_experiment.py`
- Cluster-cache raw per-task results (n=1052, 0 errors): `runtime/bfcl_cluster_cache_20260811/results.jsonl`
