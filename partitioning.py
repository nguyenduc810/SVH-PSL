import numpy as np
from pymoo.util.ref_dirs import get_reference_directions

def sampling_vector_randomly(n_obj, n_sample, n_region):
    # Sample evenly distributed n_region on n_obj space from range [0, 1], then between each region, sampling randomly n_sample reference vectors for partitioning (sampling ratio ~10%)
    if n_obj == 2:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=400)
    else:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=30)
    region = np.linspace(0, ref_dirs.shape[0], n_region + 1, dtype=np.int32)
    for i in range(n_region):
        random_index = np.random.randint(region[i], region[i+1], size=n_sample)
        index = random_index if i== 0 else np.append(index, random_index)
    vectors = ref_dirs[index]
    return vectors

def sampling_vector_evenly(n_obj, n_sample):
    # Sample evenly distributed n_sample reference vectors on n_obj space from range [0, 1]
    return get_reference_directions("energy", n_obj, n_sample, seed=1)
