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
python3 pk_example.py --gda-iterations 30000 --nparallel 100 --seed 123 --name "pk_gda"
python3 pk_example.py --gda-iterations 30000 --nparallel 100 --seed 123 --sgd --name "pk_sgd"
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
