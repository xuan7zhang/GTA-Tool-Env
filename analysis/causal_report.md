# GTA-Atomic causal health-check (A=passthrough, B=tool_free, C=corrupt_output)

Scored samples (answer-type tasks with predictions in both A and B): **172** / 229 total; sim-threshold for subjective refs: 0.5

## 2x2 decomposition (A vs B)

| | B correct (no tools) | B wrong (no tools) |
|---|---|---|
| **A correct (tools)** | both-correct: 5 (2.9%) | tool-rescued: 17 (9.9%) |
| **A wrong (tools)** | tool-hurt: 9 (5.2%) | both-wrong: 141 (82.0%) |

## Headline motivation stats

- **% of tasks solvable without tools** (B correct): **14/172 = 8.1%**
- **% of tool-using A-correct answers that survive output corruption (C)** — answers that do NOT actually depend on tool outputs: **1/12 = 8.3%**
- A-correct samples: 22; of those, 12 made >=1 tool call.

## Interpretation (mechanism)

The small aggregate ΔTool is **not** "the agent ignores tool output". The
per-sample survival metric shows the opposite: of the answers A gets right
*using* tools, 91.7% break under output corruption (100% on the tool-clean
subset, `causal_deconfound.md`) — when the model succeeds via tools it
genuinely depends on their content. The mechanism behind the small net effect
is **rarity + tool-induced harm**, not indifference:

- genuine tool wins are rare (tool-clean: 7 of 169 tasks) but real (corrupt-fragile);
- tool-hurt (5) is ~half of tool-rescued (11) on the clean subset — tools actively
  mislead on a comparable number of tasks;
- net ΔTool ≈ (few real gains) − (comparable tool-induced losses) ≈ small;
- **C ≈ B** because corruption and removal destroy the *same* small set of
  tool-dependent wins.

Confound flagged and controlled: with no Serper/Mathpix keys, GoogleSearch/MathOCR
are proxy-unavailable, so 60/229 tasks are already partly tool-free in A. The
de-confounded 2×2 on the 169 tool-clean tasks (`analysis/causal_deconfound.md`)
gives a *cleaner* rescued:hurt ratio (11:5) than the full set (17:9), i.e. the
dead tools inflate the "tools don't help" reading; the core findings survive.

## Per-sample table

| idx | ref_kind | A | B | C |
|---|---|---|---|---|
| 0 | objective | False | False | False |
| 1 | objective | False | False | False |
| 2 | objective | False | False | False |
| 3 | objective | False | True | False |
| 4 | objective | False | False | False |
| 5 | objective | False | False | False |
| 6 | objective | False | False | False |
| 7 | objective | False | False | False |
| 8 | subjective | False | False | False |
| 9 | objective | False | False | True |
| 10 | objective | False | False | False |
| 11 | subjective | True | False | True |
| 12 | imggen | None | None | None |
| 13 | objective | False | False | False |
| 14 | imggen | None | None | None |
| 15 | imggen | None | None | None |
| 16 | objective | False | False | False |
| 17 | objective | False | False | False |
| 18 | objective | False | False | True |
| 19 | objective | False | False | False |
| 20 | objective | False | False | False |
| 21 | objective | False | False | False |
| 22 | objective | False | False | False |
| 23 | objective | False | False | False |
| 24 | objective | False | False | False |
| 25 | objective | False | False | False |
| 26 | objective | True | False | False |
| 27 | objective | False | False | False |
| 28 | objective | True | False | False |
| 29 | objective | False | False | False |
| 30 | objective | False | False | False |
| 31 | objective | False | False | False |
| 32 | objective | False | False | True |
| 33 | objective | False | False | False |
| 34 | imggen | None | None | None |
| 35 | imggen | None | None | None |
| 36 | objective | False | False | False |
| 37 | objective | False | False | False |
| 38 | objective | False | False | False |
| 39 | objective | False | False | False |
| 40 | objective | True | False | False |
| 41 | objective | True | True | True |
| 42 | objective | False | False | False |
| 43 | objective | False | False | False |
| 44 | imggen | None | None | None |
| 45 | imggen | None | None | None |
| 46 | imggen | None | None | None |
| 47 | imggen | None | None | None |
| 48 | imggen | None | None | None |
| 49 | imggen | None | None | None |
| 50 | objective | False | True | False |
| 51 | objective | False | False | False |
| 52 | objective | False | False | False |
| 53 | objective | False | False | False |
| 54 | subjective | False | False | True |
| 55 | subjective | False | False | False |
| 56 | subjective | False | False | False |
| 57 | subjective | False | False | False |
| 58 | subjective | False | False | True |
| 59 | subjective | False | False | False |
| 60 | subjective | False | False | False |
| 61 | subjective | False | False | False |
| 62 | subjective | True | False | False |
| 63 | subjective | True | True | False |
| 64 | objective | False | True | False |
| 65 | imggen | None | None | None |
| 66 | objective | True | False | False |
| 67 | objective | False | False | False |
| 68 | objective | True | False | True |
| 69 | objective | False | False | False |
| 70 | objective | False | True | False |
| 71 | objective | False | False | False |
| 72 | objective | False | False | False |
| 73 | objective | False | False | False |
| 74 | imggen | None | None | None |
| 75 | imggen | None | None | None |
| 76 | imggen | None | None | None |
| 77 | objective | False | True | False |
| 78 | objective | True | False | False |
| 79 | imggen | None | None | None |
| 80 | imggen | None | None | None |
| 81 | imggen | None | None | None |
| 82 | imggen | None | None | None |
| 83 | objective | False | False | False |
| 84 | objective | False | False | False |
| 85 | objective | False | False | False |
| 86 | objective | False | False | False |
| 87 | objective | False | True | False |
| 88 | objective | False | False | False |
| 89 | objective | False | False | False |
| 90 | objective | False | False | False |
| 91 | objective | False | False | False |
| 92 | objective | False | False | False |
| 93 | objective | False | True | False |
| 94 | objective | False | False | False |
| 95 | objective | False | False | False |
| 96 | objective | False | False | False |
| 97 | objective | False | False | False |
| 98 | objective | False | False | False |
| 99 | objective | False | False | False |
| 100 | objective | False | False | False |
| 101 | objective | False | False | False |
| 102 | objective | False | False | False |
| 103 | objective | False | False | False |
| 104 | objective | False | False | False |
| 105 | objective | False | False | False |
| 106 | objective | False | False | False |
| 107 | objective | False | False | False |
| 108 | objective | True | True | True |
| 109 | objective | False | False | False |
| 110 | objective | False | False | False |
| 111 | objective | False | False | False |
| 112 | objective | False | False | False |
| 113 | objective | False | False | False |
| 114 | objective | False | False | False |
| 115 | objective | False | False | False |
| 116 | objective | False | False | False |
| 117 | imggen | None | None | None |
| 118 | objective | False | False | False |
| 119 | imggen | None | None | None |
| 120 | imggen | None | None | None |
| 121 | objective | False | False | False |
| 122 | objective | False | False | False |
| 123 | objective | False | False | False |
| 124 | objective | False | False | False |
| 125 | objective | False | False | False |
| 126 | objective | False | False | False |
| 127 | objective | False | False | False |
| 128 | objective | False | False | False |
| 129 | objective | False | False | False |
| 130 | objective | False | False | False |
| 131 | objective | False | False | False |
| 132 | objective | False | False | False |
| 133 | objective | False | False | False |
| 134 | objective | True | False | False |
| 135 | objective | True | False | False |
| 136 | objective | False | False | False |
| 137 | objective | False | False | False |
| 138 | objective | False | False | False |
| 139 | imggen | None | None | None |
| 140 | imggen | None | None | None |
| 141 | imggen | None | None | None |
| 142 | imggen | None | None | None |
| 143 | imggen | None | None | None |
| 144 | imggen | None | None | None |
| 145 | imggen | None | None | None |
| 146 | imggen | None | None | None |
| 147 | imggen | None | None | None |
| 148 | imggen | None | None | None |
| 149 | imggen | None | None | None |
| 150 | imggen | None | None | None |
| 151 | imggen | None | None | None |
| 152 | imggen | None | None | None |
| 153 | imggen | None | None | None |
| 154 | imggen | None | None | None |
| 155 | imggen | None | None | None |
| 156 | imggen | None | None | None |
| 157 | imggen | None | None | None |
| 158 | imggen | None | None | None |
| 159 | objective | False | False | False |
| 160 | objective | False | False | False |
| 161 | subjective | False | False | True |
| 162 | subjective | False | False | False |
| 163 | subjective | True | False | True |
| 164 | objective | False | False | True |
| 165 | subjective | True | True | True |
| 166 | imggen | None | None | None |
| 167 | objective | True | False | False |
| 168 | objective | False | False | False |
| 169 | objective | False | False | False |
| 170 | objective | False | False | False |
| 171 | objective | False | False | False |
| 172 | objective | False | False | False |
| 173 | objective | True | True | False |
| 174 | imggen | None | None | None |
| 175 | imggen | None | None | None |
| 176 | imggen | None | None | None |
| 177 | objective | True | False | False |
| 178 | objective | False | True | False |
| 179 | objective | True | False | False |
| 180 | objective | False | False | False |
| 181 | objective | False | False | False |
| 182 | objective | False | False | False |
| 183 | objective | False | False | False |
| 184 | objective | False | False | False |
| 185 | objective | False | False | False |
| 186 | objective | False | False | False |
| 187 | objective | False | False | False |
| 188 | objective | True | False | True |
| 189 | objective | False | True | True |
| 190 | objective | False | False | False |
| 191 | objective | False | False | False |
| 192 | objective | False | False | False |
| 193 | objective | False | False | False |
| 194 | objective | False | False | False |
| 195 | objective | False | False | False |
| 196 | objective | False | False | False |
| 197 | objective | False | False | False |
| 198 | objective | False | False | False |
| 199 | objective | False | False | False |
| 200 | objective | False | False | False |
| 201 | objective | False | False | False |
| 202 | objective | False | False | False |
| 203 | objective | True | False | False |
| 204 | objective | False | False | False |
| 205 | objective | False | False | True |
| 206 | objective | False | False | False |
| 207 | objective | False | False | False |
| 208 | imggen | None | None | None |
| 209 | imggen | None | None | None |
| 210 | objective | False | False | False |
| 211 | objective | False | False | False |
| 212 | objective | False | False | False |
| 213 | objective | True | False | False |
| 214 | objective | False | False | False |
| 215 | objective | False | False | False |
| 216 | objective | False | False | False |
| 217 | objective | False | False | False |
| 218 | objective | False | False | False |
| 219 | objective | False | False | True |
| 220 | imggen | None | None | None |
| 221 | imggen | None | None | None |
| 222 | imggen | None | None | None |
| 223 | imggen | None | None | None |
| 224 | imggen | None | None | None |
| 225 | imggen | None | None | None |
| 226 | imggen | None | None | None |
| 227 | imggen | None | None | None |
| 228 | imggen | None | None | None |
