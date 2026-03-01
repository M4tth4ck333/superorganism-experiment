# Mycelium Economic Simulation

Simulates the economic lifecycle of self-replicating mycelium nodes: income (faucet), expenses (rent), reproduction (spawning children), and death (running out of funds).

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m host.simulator -c config/small.yaml
```

Output goes to `data/events.csv`. Set `tick_interval: 0` in the config to run at full speed.

## Changing parameters

All parameters live in `config/default.yaml`. To experiment, change it or create a new config file.
```bash
python -m host.simulator -c config/<your_file>.yaml
```

## How a tick works

Each tick is a synchronous round:

1. Rent is deducted from every living node
2. Bankrupt nodes are killed
3. Faucet injects funds
4. Every living node decides exactly once: `none`, `spawn`, or `failsafe`
5. Decisions are processed (spawns transfer inheritance; failsafe nodes donate all funds then die)
6. Conservation invariant is checked

## Plotting results

```bash
python analysis/plot.py data/events.csv -o data/plots
```
