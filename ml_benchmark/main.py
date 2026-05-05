# https://docs.nvidia.com/deeplearning/dali/archives/dali_140/user-guide/docs/examples/general/data_loading/numpy_reader.html

import os
import numpy as np
from glob import glob
import shutil
import tempfile
from nvidia.dali import pipeline_def, fn

batch_size = 4
dali_extra_dir = os.environ['DALI_EXTRA_PATH']
data_dir_2d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_2d_slices', 'STU00001')
data_dir_3d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_3d', 'STU00001')


