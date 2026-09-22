# SSBI Phase 3 Summary

Generated: 2026-09-01T09:31:30.941006 UTC

## Samples
- Genuine 10
- Forged 10
- Total 20

## Extraction
- Success 20/20
- Clean 15 Acceptable 5 Contaminated 0 Failed 0
- Mean quality genuine 0.7601 forged 0.7593 overall 0.7597
- Mean IoU 0.7317 median 0.8101 >=0.30 20 >=0.50 15

## Genuine Scores
- [0.1844, 0.1791, 0.1415, 0.1592, 0.2388, 0.1160, 0.1569, 0.1293, 0.1569, 0.1834]
- Mean 0.1646 Median 0.1581 Std 0.0328 Min 0.1160 Max 0.2388 Q1 0.1454 Q3 0.1823

## Forged Scores
- [0.2226, 0.3065, 0.2384, 0.2097, 0.2001, 0.2246, 0.2371, 0.2264, 0.2195, 0.1918]
- Mean 0.2277 Median 0.2236 Std 0.0298 Min 0.1918 Max 0.3065 Q1 0.2122 Q3 0.2344

## Separation
- Mean diff -0.0631 Median diff -0.0655
- ROC AUC 0.0900 Pairwise G>F 9/100 ties 0
- Lowest genuine 0.1160 Highest forged 0.3065 Overlap YES

## Frozen Policy L=0.0895 U=0.6898
- Genuine: Conforme 0 Contrôle manuel 10 Non conforme 0
- Forged: Conforme 0 Contrôle manuel 10 Non conforme 0

## Database
- Cheques for SSBI: 20
- VR: 20 (after fix; before fix 19 due to CHECK violation on S7_GENUINE_03)
- Total VR in DB: 25 (including 5 historical CEDAR/other)
- Total SC in DB: 105 (20*5 +5 historical)
- Note: S7_GENUINE_03.png (Ref6 -0.012173) previously failed CHECK 0<=score<=1; after fix AllowNegativeCosineSimilarity persisted as VerificationId 44 with 5 SC rows including -0.0122

## Notes
- Forged 09/10 reuse 2 forgery samples on different templates per README (8 unique forgeries, S7_FORGED_09 reuses F1#33 on check_002, S7_FORGED_10 reuses F1#34 on check_003)
- All samples processed via ChequeService, VerificationService, AI V2 mean raw cosine, no threshold change
- Extraction V2.4 remained frozen; DB fix now correctly supports raw cosine [-1,1] (previously 0..1)
