# Bonner Lab | Datasets

You can find extensive documentation on the [`bonner-datasets` website](https://bonnerlab.github.io/bonner-datasets/)!

## Quickstart

1. Set the environment variables `BONNER_BRAINIO_CACHE` to `/data/shared/brainio`.
2. Create a BrainIO Catalog:

    ```python
    from pathlib import Path

    CATALOG = Catalog(
        identifier="bonner-datasets",
        csv_file=Path(<path to catalog.csv in this repo>),
        cache_directory=None,
    )
    ```

### Natural Scenes Dataset

Open the data for a subject.

```python
from bonner.datasets.allen2021_natural_scenes import open_subject_assembly

filepath = CATALOG.load_data_assembly(identifier="allen2021.natural_scenes", check_integrity=False)
assembly = open_subject_assembly(subject, filepath=filepath, **kwargs)
```

Use the other utility functions in `bonner.datasets.allen2021_natural_scenes` to e.g. average betas across repetitions of stimuli.
