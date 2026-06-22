# Audio→semantic approach comparison

samples: basketball_01, football_10  |  AST available: True


## basketball_01  (18 events)
- detector counts: onset_strength=16, spectral_flux=15, hf_transient=17, combined=18
- detector overlap: onset_strength∩spectral_flux=1.0, onset_strength∩hf_transient=0.94, spectral_flux∩hf_transient=1.0
- visual grounding rate: **1.0**  (cues: vel_reversal=17, proximity_min=1)
- classifier label dist: rule: rattle=9, bounce=9 | cluster: bounce=16, rattle=2 | pretrained: strike=17, bounce=1
- classifier agreement: rule~cluster=0.5, rule~pretrained=0.0, cluster~pretrained=0.06
- cluster silhouette: 0.401

## football_10  (17 events)
- detector counts: onset_strength=17, spectral_flux=9, hf_transient=8, combined=17
- detector overlap: onset_strength∩spectral_flux=1.0, onset_strength∩hf_transient=1.0, spectral_flux∩hf_transient=1.0
- visual grounding rate: **0.12**  (cues: none=15, flow_spike=2)
- classifier label dist: rule: tap=9, strike=8 | cluster: tap=10, strike=7 | pretrained: unknown=9, strike=4, bounce=3, tap=1
- classifier agreement: rule~cluster=0.82, rule~pretrained=0.12, cluster~pretrained=0.18
- cluster silhouette: 0.281

## Aggregate (generalization)
- cluster~pretrained agreement across samples: mean=0.12 (min=0.06, max=0.18)
- rule~cluster agreement across samples: mean=0.66 (min=0.5, max=0.82)
- rule~pretrained agreement across samples: mean=0.06 (min=0.0, max=0.12)
- grounding rate across samples: mean=0.56 (min=0.12, max=1.0)

Interpretation: a higher grounding rate + higher cross-sample classifier agreement = a more trustworthy, generalizable event source for the loss.
