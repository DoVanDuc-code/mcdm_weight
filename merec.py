import numpy as np
from objective_weighting.mcda_methods import VIKOR
from objective_weighting import weighting_methods as mcda_weights
from objective_weighting import normalizations as norms
from objective_weighting.additions import rank_preferences

matrix = np.array([[4, 3, 3, 3, 5],
                   [4, 4, 4, 4, 5],
                   [2, 3, 4, 4, 4],
                   [2, 5, 5, 5, 3],
                   [1, 5, 5, 4, 3]])

types = np.array([-1, 1, 1, 1, 1])
weights = mcda_weights.merec_weighting(matrix,types)

# Create the VIKOR method object
vikor = VIKOR(normalization_method=norms.minmax_normalization)
# Calculate alternatives preference function values with VIKOR method
pref = vikor(matrix, weights, types)
# Rank alternatives according to preference values
ranking = rank_preferences(pref, reverse = False)

print(weights)


