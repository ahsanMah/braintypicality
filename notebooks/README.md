## Outline

These notebooks can be used to reproduce the experiments that were part of the dissertation document.

### Prerequisite
1. Install `sade`
2. Install the `simpsom` library
    ```bash
    git clone https://github.com/ahsanMah/simpsom
    cd simpsom
    python setup.py install --user
    ```
    - Note that my version of the library is modified to be more deterministic and will give reproducible results when using PCA inititialization

2. Ensure that the score norms were computed using `sade`
3. Ensure the flow likelihoods were computed using `sade`

### Order of operations

1. Run the `down_syndrome_som.ipynb` notebook with the chose `som_height` and `som_width`
    - This will create a CSV file with sample IDs and the matching BMU (prototype number)
    - The notebook will also create some interactive plots
2. Run `parcellation_analysis.ipynb`
    - This notebook will ingest the likelihood scores and compute the per-parcellation likelihoods for each cohort
    - By default the AAL+CSF atlas is used which combines the AAL parcellation with an automated CSF estimation (using AntsPyNet)
3. Run `roi_correlation_analysis.ipynb`
    - This will produce all the plots for the correlation analyses
    - The DS prototype analysis starts at Section `Analyzing DS-Prototype Samples`
    - Note that the `run_correlation_analysis` function will perform 10,000 resampling steps across 1480 correlation tests (assuming the default 13x13 prototype). This can take 3-4 hours to run.