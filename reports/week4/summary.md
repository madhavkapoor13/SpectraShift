# Week 4 implementation summary

Date: 14 September 2026  
Stage: full M2-M4 VICReg pretraining  
Status: complete; Week 5 approved

## Frozen matrix

- M2: RGB bands B04/B03/B02, no spectral dropout.
- M3: ten core bands, no spectral dropout.
- M4: ten core bands, asymmetric one-group dropout with probability 0.25.
- Seeds: 17, 29, and 43.
- Shared schedule: AdamW, learning rate `1e-4`, batch 64, 5-epoch warmup, cosine decay to `1e-6`, and 60 epochs.

## Completion contract

Each run must finish 18,720 optimizer steps with finite diagnostics, zero AMP overflows, projector standard deviation at least 0.5, and projector effective rank at least 32. It must retain a verified epoch-60 training state and a finite encoder-only export. Week 5 remains blocked until the aggregate verifier confirms all nine unique encoders.

## Kaggle attempt 1

The first seed-17 run completed the 60 M2 epochs but failed the exact-step gate after AMP scale growth caused at least one optimizer-step skip. Week 4 now keeps the Week 3-proven scale at 128 for the entire run, stops immediately on any overflow, and prunes recovery checkpoints only after every stability and completion gate passes. The failed output is not a valid Week 4 checkpoint and is not used for recovery.

## Seed 17 verified result

The corrected job completed M2, M3, and M4 with exactly 18,720 optimizer steps each and zero AMP overflows. All checkpoint and encoder hashes match, all 1,440 logged diagnostic records are finite, and no evaluation labels were loaded. Projector standard deviation/effective rank were `0.9891/59.90` for M2, `0.9891/58.85` for M3, and `0.9877/58.10` for M4. Total measured GPU time was 1.9566 hours.

## Seed 29 verified result

The seed-29 job completed M2, M3, and M4 with exactly 18,720 optimizer steps each and zero AMP overflows. All checkpoint and encoder hashes match, all 1,440 logged diagnostic records are finite, and no evaluation labels were loaded. Projector standard deviation/effective rank were `0.9892/57.44` for M2, `0.9887/57.52` for M3, and `0.9890/57.61` for M4. Total measured GPU time was 2.0036 hours.

## Seed 43 verified result

The seed-43 job completed M2, M3, and M4 with exactly 18,720 optimizer steps each and zero AMP overflows. All checkpoint and encoder hashes match, all 1,440 logged diagnostic records are finite, and no evaluation labels were loaded. Projector standard deviation/effective rank were `0.9877/57.98` for M2, `0.9893/58.73` for M3, and `0.9874/59.40` for M4. Total measured GPU time was 2.0128 hours.

## Aggregate verification

The CPU aggregate verified all nine expected model-seed combinations and nine unique encoder hashes against the three retained seed summaries. Every run completed exactly 18,720 optimizer steps, passed its stability, completion, and compute gates, recorded zero AMP overflows, and excluded evaluation labels. Total measured Week 4 GPU time was 5.9730 hours. The aggregate reports `week4_complete: true` and `week5_approved: true`.
