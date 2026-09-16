# Protocol amendment 001: Portugal geographic groups

Date: 13 September 2026  
Status: adopted before training

## Trigger

The verified BigEarthNet v2 clean metadata contains 54 MGRS tiles. Portugal occupies three (`T29SNB`, `T29SNC`, and `T29SND`), fewer than the provisional target of five geographic groups per evaluation partition. Finland occupies 21 MGRS tiles. This finding was made before pixel staging, model training, or target-label inspection.

## Resolution

Finland and Portugal remain the fixed, fully held-out target countries. Week 2 will derive exact patch footprints in EPSG:3035 and assign target patches to fixed 12 km spatial blocks. These blocks will be used for target-domain group counts and geographic bootstrap resampling. MGRS tiles remain the higher-level acquisition/geography field reported with every result.

The 12 km block origin will be fixed at `(0 m, 0 m)` in EPSG:3035. A patch belongs to the block containing its centroid. Repeated acquisitions of the same footprint remain in one block. No block assignment will use labels or model predictions.

## Consequences

- The country holdout is unchanged.
- No Portugal sample can enter U, D, V, or I.
- The amendment increases the number of independent resampling units but does not turn them into independent countries.
- Claims remain limited to transfer into the two selected countries.
- Training remains blocked until exact raster footprints and the 2.4 km source/evaluation buffer pass Week 2 checks.

