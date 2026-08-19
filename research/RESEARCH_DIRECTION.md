# Research direction: Federated + Reliable ABR

This branch establishes experimental infrastructure for evaluating learned ABR
policies. It does not claim a completed research contribution.

The working hypothesis is that heterogeneous streaming clients could
collaboratively learn a shared ABR policy, while each client estimates whether
that policy is competent under its current network conditions. A client judged
untrusted could use a robust classical fallback such as BOLA or MPC.

```text
heterogeneous clients
        |
federated learned ABR policy
        |
local policy competence / trust estimation
       / \\
 trusted  untrusted
    |        |
 learned  classical fallback
 policy    (BOLA/MPC)
```

None of federation, competence estimation, trust routing, BOLA, or MPC is
implemented in this foundation. They should be added only after controlled
baseline and distribution-shift evidence exists.
