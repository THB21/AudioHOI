# Ping-pong ablation evidence

This directory stores case-specific geometric and physical diagnostics for the
ping-pong wall sequence. They supplement, rather than replace, the repository's
unified ablation metrics in
`output/pingpong_unified_ablation_evaluation/ablation_table.csv`.

The frozen pose paths and SHA-256 hashes in `case_specific_metrics.json` bind
every number to its Full, No-VLM, or No-audio trajectory. The diagnostics show
two distinct gains: VLM rejects the duplicated-ball visual branch and prevents
large non-contact reversals, while audio supplies the impact windows needed for
valid paddle/wall collision reversal and near-zero paddle-contact gap.

The final joint object/human review render remains at
`samples_known_object/14_pingpong_wall/results/renders/pingpong_planar_pnp_recomputed_contact_with_human/`.
