# 1D Bayes Filter

Discrete Bayes filtering for a robot moving through a 17-cell one-dimensional grid world. The implementation includes:

- a probabilistic forward/backward motion model with boundary handling
- a binary floor-color sensor model
- recursive belief updates for known or unknown initial position

## Run

```bash
python - <<'PY'
import numpy as np
import ex1

belief = np.zeros(17)
belief[7] = 1.0
world = np.loadtxt("world.data", delimiter=",", dtype=int)
observations = np.loadtxt("observations.data", delimiter=",", dtype=int)

final_belief = ex1.recursive_bayes_filter(list("FFFFBBFFB"), observations, belief, world)
print("most likely cell:", np.argmax(final_belief))
print("probability:", final_belief.max())
PY
```
