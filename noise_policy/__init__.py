"""A policy over FastWAM's initial noise: pi(eps | o, l, s) = N(mu, diag(sigma^2)).

FastWAM turns noise into an action chunk with a fixed, deterministic denoising chain. Instead
of drawing that noise from N(0, I), a small network conditioned on the observation chooses
the Gaussian it is drawn from; FastWAM itself stays frozen and unchanged.

    config.py     NoisePolicyConfig
    modeling.py   NoisePolicy (the architecture) and NoiseDistribution (what it outputs)
    fastwam.py    running FastWAM from a given noise, and building the policy's inputs
"""
