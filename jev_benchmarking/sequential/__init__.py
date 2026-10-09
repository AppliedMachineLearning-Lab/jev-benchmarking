"""Sequential decision benchmarks: one decision request per step, the next request depends on the answer.

The single-step suite sends all requests up front. Here a decision model acts in an environment: at every
step the environment builds a request (state + typed questions), the model answers, the chosen option is
executed, and the episode is scored at the end. Requests are still cached by their exact JSON, so a rerun
of a deterministic episode replays from the cache without paying again.

Layout:
- `backends.py`: who answers a request (Jev over the API, an open-weight model, or a scripted policy).
- `envs/`: the environments (Blackjack with exact values, ALFWorld).
- `runner.py`: plays episodes and writes one JSON line per episode.
- `metrics.py`: success, cost, and calibration of the success estimates against outcomes.
"""
