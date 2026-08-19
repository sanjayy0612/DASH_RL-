# Hypotheses

## H0 — Baseline

PPO performance is stable when training and evaluation network distributions are similar.

## H1 — Distribution shift

PPO performance degrades under network conditions that differ materially from its training distribution.

## H2 — Controller complementarity

There may exist network regimes where a classical ABR controller outperforms the learned controller.

## H3 — Trust routing (future; not implemented)

Information about policy competence may allow a system to choose between learned and classical controllers more reliably than always using one controller.

## H4 — Federation (future; not implemented)

Heterogeneous clients may provide useful information not only for learning a shared policy but eventually for learning where that policy is reliable.

These are testable hypotheses, not claims of established results or novelty.
