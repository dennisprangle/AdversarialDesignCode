## Adversarial design code

The repository is code used in the paper "Bayesian experimental design without posterior calculations: an adversarial approach" (https://arxiv.org/abs/1904.05703).

To reproduce the results in the paper run

```bash
python3 pk_example.py --gda-iterations 30000 --nparallel 100 --seed 123 --name "pk_gda"
python3 pk_example.py --gda-iterations 30000 --nparallel 100 --seed 123 --sgd --name "pk_sgd"
python3 pk_plots.py
```
