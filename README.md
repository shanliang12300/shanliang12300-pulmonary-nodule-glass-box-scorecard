# From black box to glass box: interpretable pulmonary nodule risk assessment by symbolic rule compression

Analysis code for the manuscript. The pipeline reproduces:
quality control and cohort split (common_preprocessing) -> CT report
text-mining (radiology_01-03) -> blood feature selection (fig2) ->
joint blood+CT selection and model comparison (fig3) -> symbolic rule
compression and scorecard derivation (fig4) -> external validation and
the desktop tool (fig5).

## Requirements
Python 3.10.8; see requirements.txt.

## Data
Patient-level data contain sensitive clinical information and are
available from the corresponding author upon reasonable request.
Frozen model artefacts (rule weights, reference intervals) are in frozen/.
