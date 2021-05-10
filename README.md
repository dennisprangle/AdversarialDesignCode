## Adversarial design code

The repository is code used in the paper "Bayesian experimental design without posterior calculations: an adversarial approach" (https://arxiv.org/abs/1904.05703).

To reproduce the results in the paper run

```bash
python3 poisson_example.py --lr-a 1e-3 --name "poisson3"
python3 poisson_example.py --lr-a 1e-4 --name "poisson4"
python3 poisson_example.py --lr-a 1e-5 --name "poisson5"
python3 poisson_plots.py
```

```bash
python3 pk_example.py --gda-iterations 100000 --nsamples 100 --nparallel 100 --seed 123 --point-exchange --name "pk_gda_K100"
python3 pk_example.py --gda-iterations 100000 --nsamples 10 --nparallel 100 --seed 123 --point-exchange --name "pk_gda_K10"
python3 pk_example.py --gda-iterations 100000 --nsamples 1 --nparallel 100 --seed 123 --point-exchange --name "pk_gda_K1"
python3 pk_example.py --gda-iterations 100000 --nsamples 1 --nparallel 100 --seed 123 --sgd --point-exchange --name "pk_sgd"
python3 pk_example.py --gda-iterations 100000 --nsamples 1 --nparallel 100 --seed 123 --gaps --name "pk_gaps"
python3 pk_plots.py
python3 pk_posteriors.py
```

```bash
python3 geostats_example.py --length-scale 0.01 --name "geo1"
python3 geostats_example.py --length-scale 0.02 --name "geo2"
python3 geostats_example.py --length-scale 0.04 --name "geo3"
python3 geostats_example.py --length-scale 0.08 --name "geo4"
python3 geostats_plots.py
```

There is a R subdirectory to run a comparison analysis using the `acebayes` package. This can be run using the following command. Note you should be in the R directory when this is run. Also, this analysis takes roughly 3 days to run.
```bash
R CMD BATCH --no-save PK_SIG.R
```
After transferring the csv file produced to the `outputs` folder, plot for the paper can be produced using
```bash
python3 pk_plots_R.py
```

Finally, to produce plots for the methods of Foster et al (produced using code in [https://github.com/dennisprangle/pyro/tree/sgboed-reproduce]), transfer its results to the outputs folder and run
```bash
python3 pk_plots_foster.py
```

