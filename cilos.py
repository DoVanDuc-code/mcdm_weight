import numpy as np
from objective_weighting.mcda_methods import VIKOR
from objective_weighting import weighting_methods as mcda_weights
from objective_weighting import normalizations as norms
from objective_weighting.additions import rank_preferences

matrix =np.array([
    [5, 8, 4],
    [7, 6, 8],
    [8, 8, 6],
    [7, 4, 6]
], dtype='int')-
 # data = norms()
types = np.array([1, 1, 1])
try:
    weights = mcda_weights.cilos_weighting(matrix, types)
except np.linalg.LinAlgError as e:
    print(f"An error occurred while calculating weights: {str(e)}")
    weights = None

if weights is not None:
    # Create the VIKOR method object
    vikor = VIKOR(normalization_method=norms.minmax_normalization)
    # Calculate alternatives preference function values with VIKOR method
    pref = vikor(data, weights, types)
    # Rank alternatives according to preference values
    ranking = rank_preferences(pref, reverse=False)
    print(weights)