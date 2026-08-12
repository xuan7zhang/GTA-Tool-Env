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

| Cluster | n tasks | distinct GT tools | ratio | Coherence | Top tools |
|---:|---:|---:|---:|---|---|
| 26 | 122 | 97 | 0.80 | **incoherent** | `multiply`, `add`, `sub`, `fahrenheit_to_celsius`, `celsius_to_fahrenheit` |
| 8 | 62 | 3 | 0.05 | coherent | `Media_3_FindMovies`, `Media_3_PlayMovie`, `Music_3_PlayMedia` — media/music playback |
| 2 | 60 | 4 | 0.07 | coherent | `Movies_1_BuyMovieTickets`, `Movies_1_FindMovies`, `Movies_1_GetTimesForMovie` — movie tickets |
| 7 | 57 | 3 | 0.05 | coherent | `Events_3_FindEvents`, `Events_3_BuyEventTickets`, `Movies_3_FindMovies` — events |
| 9 | 48 | 4 | 0.08 | coherent | `Restaurants_2_ReserveRestaurant`, `Restaurants_2_FindRestaurants` — dining |
| 30 | 47 | 4 | 0.09 | coherent | `Payment_1_RequestPayment`, `Payment_1_MakePayment`, `Restaurants_2_ReserveRestaurant` — payments |
| 13 | 45 | 3 | 0.07 | coherent | `Homes_2_FindHomeByArea`, `Homes_2_ScheduleVisit`, `RentalCars_3_GetCarsAvailable` — home search |
| 25 | 38 | 3 | 0.08 | coherent | `Music_3_PlayMedia`, `Music_3_LookupMusic`, `Weather_1_GetWeather`, `RideSharing_2_GetRide` — rideshare/misc |
| 14 | 38 | 2 | 0.05 | coherent | `get_service_providers`, `view_service_provider_profile` — service marketplace |
| 29 | 36 | 4 | 0.11 | coherent | `Hotels_4_ReserveHotel`, `Hotels_4_SearchHotel`, `Travel_1_FindAttractions` — hotel/travel |
| 18 | 33 | 12 | 0.36 | mixed | `get_current_weather`, `add_postgres_server`, `dartfx_help` — dev-tools grab-bag |
| 5 | 33 | 8 | 0.24 | mixed | `add_postgres_server`, `add_mtnards_server`, `dartfx_help` — server config |
| 0 | 32 | 3 | 0.09 | coherent | `Flights_4_SearchOnewayFlight`, `RentalCars_3_GetCarsAvailable` — flights/cars |
| 6 | 29 | 4 | 0.14 | coherent | `Alarm_1_GetAlarms`, `Alarm_1_AddAlarm`, `Services_1_BookAppointment` — alarms/appointments |
| 27 | 27 | 3 | 0.11 | coherent | `Buses_3_FindBus`, `Buses_3_BuyBusTicket`, `Events_3_FindEvents` — buses/events |
| 16 | 26 | 3 | 0.12 | coherent | `Flights_4_SearchOnewayFlight`, `Hotels_2_BookHouse` — flights/hotels |
| 32 | 25 | 2 | 0.08 | coherent | `Hotels_2_BookHouse`, `Hotels_2_SearchHouse`, `Travel_1_FindAttractions` — home rentals |
| 10 | 21 | 2 | 0.10 | coherent | `Services_4_BookAppointment`, `Services_4_FindProvider`, `Weather_1_GetWeather` |
| 1 | 21 | 1 | 0.05 | coherent | `Movies_3_FindMovies`, `Music_3_PlayMedia`, `Music_3_LookupMusic` |
| 23 | 17 | 5 | 0.29 | mixed | `OpenWeatherMap.get_current_weather`, `ControlAppliance.execute`, `HNA_NEWS.search` — smart-home grab-bag |
| 4 | 17 | 1 | 0.06 | coherent | `Buses_3_FindBus`, `Buses_3_BuyBusTicket`, `Events_3_FindEvents` |
| 33 | 16 | 8 | 0.50 | mixed | `uber.ride`, `uber.eat.order`, `flight.status.check` |
| 11 | 16 | 5 | 0.31 | mixed | `set_alarm`, `set_volume`, `play_song` |
| 15 | 15 | 4 | 0.27 | mixed | `get_sensor_readings_latest`, `get_sensor_readings_history_by_interval` — IoT sensors |
| 22 | 15 | 5 | 0.33 | mixed | `get_detail_adriel_project`, `get_adriel_list_projects`, `get_adriel_profile` — portfolio API |
| 21 | 15 | 2 | 0.13 | coherent | `version_api.VersionApi.get_version`, `project_api...` — dev/package-registry API |
| 20 | 15 | 1 | 0.07 | coherent | `Media_3_FindMovies`, `Media_3_PlayMovie`, `Weather_1_GetWeather` |
| 19 | 14 | 4 | 0.29 | mixed | `inventory_management`, `product_search`, `order_status_check` — e-commerce |
| 12 | 13 | 6 | 0.46 | mixed | `detail_project`, `detail_experience_and_education`, `list_projects` — portfolio API |
| 31 | 13 | 1 | 0.08 | coherent | `Events_3_FindEvents`, `Events_3_BuyEventTickets`, `Hotels_4_ReserveHotel` |
| 3 | 12 | 2 | 0.17 | mixed | `Trains_1_GetTrainTickets`, `Trains_1_FindTrains`, `Hotels_2_BookHouse` |
| 34 | 11 | 4 | 0.36 | mixed | `analysis_api...`, `acl_api...` — internal platform API |
| 17 | 11 | 3 | 0.27 | mixed | `Trains_1_GetTrainTickets`, `Trains_1_FindTrains`, `Payment_1_RequestPayment` |
| 36 | 10 | 4 | 0.40 | mixed | `stock_price.get`, `weather.get`, `weather.get_weather` |
| 28 | 10 | 1 | 0.10 | coherent | `RideSharing_2_GetRide`, `Travel_1_FindAttractions`, `Weather_1_GetWeather` |
| 35 | 9 | 3 | 0.33 | mixed | `CustomDashboardsApi...`, `api_token_api...` — dashboards platform API |
| 37 | 8 | 5 | 0.62 | **incoherent** | `detail_adriel_project`, `adriel_detail_experience_and_education` |
| 38 | 7 | 4 | 0.57 | **incoherent** | `search_engine.query`, `generate_image`, `generate_human_image` |
| 24 | 5 | 2 | 0.40 | mixed | `generate_image_tool`, `tts_tool`, `write_markdown_tool` |
| 39 | 3 | 2 | 0.67 | **incoherent** | `get_pods`, `get_services` — Kubernetes-ish, too few tasks to tell |

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

This has **not been tested for accuracy** — it's a structural observation
about the dataset, not a result about whether the two-tier design actually
preserves the pool-subset experiment's 93.2% accuracy at lower cost. That
would be the natural next experiment.

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

## Artifacts

- Code: `experiments/bfcl_token_likelihood/cluster_tasks_by_tool_space.py`
- Full cluster assignments (all 40, with task IDs): `runtime/bfcl_live_multiple_data/kmeans_k40_clusters.json`
