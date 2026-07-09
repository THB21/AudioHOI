"""Generic audio→semantic pipeline for HOI.

Extract audio from a video, detect acoustic events, describe them acoustically,
ground them in the surrounding video frames, and emit one generic semantic sheet
that downstream pose/contact optimization can consume. See ``audio_pipeline_design.md``.
"""
