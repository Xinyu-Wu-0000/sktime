import posixpath

import h5py

from sktime.classification.distance_based import ShapeDTW
from sktime.utils.estimator_checks import check_estimator

# check_estimator(ShapeDTW, raise_exceptions=True)


p = h5py.File("fdsf")

# p = posixpath.abspath("model")
print(p / "keras/")
