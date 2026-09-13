# Post-conversion population audit v1

This read-only gate audits the final LeRobot v2.1 exports after conversion and
before normalization/statistics/model training.

Each audited episode bundle must contain all eight model-state profiles:

- rotvec_principal × {no_wrench, full}
- rotvec_continuous × {no_wrench, full}
- quaternion × {no_wrench, full}
- rotation6d × {no_wrench, full}

The audit checks each individual export with the production validator and then
independently verifies cross-profile equivalence: exact 7D actions, timestamps
and indices, exact non-orientation state channels, exact wrench channels for all
full profiles, the same physical orientation after decoding, matching tasks and
stable provenance, and byte-identical physical camera videos. Every profile,
including the historical `rotvec_principal/full` default, must declare an exact
`model_state_profile` object in `meta/export_provenance.json`; missing profile
metadata is a blocking audit failure.

For one bundle:

```bash
scripts/run_post_conversion_audit.sh BUNDLE_ROOT OUTPUT_DIR --expected-bundles 1
```

For a population, point `INPUT_ROOT` at a directory whose direct children are
complete eight-profile bundles. After accepted episodes are frozen, write the
exact `source_processed_episode` strings one per line and pass
`--expected-sources-file` to fail closed on missing, duplicate, or unexpected
sources.

This audit runs on exported LeRobot data. It does not require OpenPI/ForceVLA
normalization, padding, tokenization, or a model environment.
