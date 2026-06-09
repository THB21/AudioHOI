# Basketball sample

This folder is kept as a sample/result holder. The old shared-camera basketball workflow scripts have been cleaned out of `scripts/shared/`.

For the active radius-free object pipeline, use:

```text
scripts/shared/radius_free_proxy/
```

If this basketball sample is used as an object-proxy/radius-free test case, generate audio events with:

```bash
python scripts/shared/radius_free_proxy/stage0_preprocess/align_audio_events.py --sample-dir samples/basketball_01
```

Existing generated outputs may still remain under `samples/basketball_01/results/`.
