# Week 8 implementation summary

Date: 18 September 2026

Stage: frozen I/Finland/Portugal evaluation

Status: implemented; Kaggle execution pending

The pre-label Week 7 state is frozen at commit `e9af130134d7b323a5f4315338d462d01cb136e7` and annotated tag `week7-complete`. Week 8 requires the exact tracked Week 7 summary hash `7c3c33fc6bb254d58565fa5cbe98bb799f2b6ff4630e91a9a9f26659ab2f273d` and checkpoint-ledger hash `fbf0a98618da16efad73d66f11f81af4a784d17dfbf7b59f52f023d25a5282f3` before label access.

The sealing stage joins final Week 2 IDs to the private candidate-label table, freezes 3,000 I, 4,000 Finland, and 4,000 Portugal rows, records 16 source-supported classes and all 19 reporting classes, and freezes MGRS and 12 km uncertainty groups. A local dry run against the real 12,500 candidate rows selected the exact 11,000 final IDs and excluded the 1,500 unused candidates.

Each GPU notebook evaluates 37 frozen checkpoints for one seed: 30 M0-M4 states, one M1RGB control, and six M5/M6 states. Predictions contain patch IDs and 19 finite logits only. The CPU aggregate verifies all 111 checkpoints and 333 domain prediction artifacts, recomputes domain and per-class metrics, calculates equal-country OOD mAP, normalized log-label AULC, paired MGRS bootstrap intervals, and 12 km sensitivity intervals, and may approve Week 9 without implementing it.

The complete local suite passes 73 tests. Week 9 code is absent by design.
