# Dataset setup

Download these processed CSV files from the Mendeley Data source and place them in this directory:

- `sound_features.csv` — 100 trials and 260 numeric acoustic features, plus a sound-segment identifier.
- `force_features.csv` — 100 trials with measured target columns `Ft`, `Fr`, and `Fa`, plus a force-file identifier.
- `cutting_parameters.csv` — 100 rows containing `cutting_speed`, `cutting_depth`, and `feed_rate`.

**Dataset landing page:** https://data.mendeley.com/datasets/y89s8vn8r4/1

The raw audio archive is not required for this processed-feature workflow.

## Alignment note

The parameter CSV has no explicit trial-ID column. The supplied row order is used to align parameter rows with the sound and force trials. The training script verifies that sound and force identifiers are sequential from 001 to 100, but it cannot independently verify the parameter-row mapping. Confirm this mapping against the source documentation before making scientific or production claims.

Check the dataset's current licence and attribution requirements before redistributing its files. The raw/processed CSVs are not mirrored in this repository.
