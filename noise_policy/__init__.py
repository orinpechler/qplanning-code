"""A policy over FastWAM's initial noise: pi(eps | o, l, s) = N(mu, diag(sigma^2)).

FastWAM turns noise into an action chunk with a fixed, deterministic denoising chain. Instead
of drawing that noise from N(0, I), a small network conditioned on the observation chooses
the Gaussian it is drawn from; FastWAM itself stays frozen and unchanged.

    config.py          NoisePolicyConfig, and the training configs
    modeling.py        NoisePolicy (the architecture) and NoiseDistribution (what it outputs)
    fastwam.py         running FastWAM from a given noise, and building the policy's inputs
    losses.py          the importance-weighted and the flow-reversal likelihood losses
    data.py            training states: frames of the demonstrations
    training.py        the loop both training scripts share (replay buffer, W&B, checkpoints)
    train_iw.py        training principle 1: Q-based importance weights
    train_reversal.py  training principle 2: Q-improved actions, flow-reversed to noise
"""
