import numpy as np
from objective_weighting.mcda_methods import VIKOR
from objective_weighting import weighting_methods as mcda_weights
from objective_weighting import normalizations as norms
from objective_weighting.additions import rank_preferences

matrix = np.array([
    [4, 4],
    [1, 5],
    [3, 2],
    [4, 2]
], dtype='float')

types = np.array([1, 1])
weights = mcda_weights.entropy_weighting(matrix)

# Create the VIKOR method object
vikor = VIKOR(normalization_method=norms.minmax_normalization)
# Calculate alternatives preference function values with VIKOR method
pref = vikor(matrix, weights, types)
# Rank alternatives according to preference values
ranking = rank_preferences(pref, reverse = False)

print(weights)