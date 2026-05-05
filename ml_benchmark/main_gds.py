# https://docs.nvidia.com/deeplearning/dali/archives/dali_140/user-guide/docs/examples/general/data_loading/numpy_reader.html

import os
import numpy as np
from glob import glob
import shutil
import tempfile
from nvidia.dali import pipeline_def, fn

batch_size = 1000
dali_extra_dir = os.environ['DALI_EXTRA_PATH']
data_dir_2d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_2d_slices', 'STU00001')
data_dir_3d = os.path.join(dali_extra_dir, 'db', '3D', 'MRI', 'Knee', 'npy_3d', 'STU00001')

data_dir = os.path.join(data_dir_2d, 'SER00001')
files = sorted([f for f in os.listdir(data_dir) if '.npy' in f])

# gds reading
@pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
def pipe_gds():
    data = fn.readers.numpy(device='gpu', file_root=data_dir, files=files)
    return data

p = pipe_gds()
p.build()
pipe_out = p.run()

data_gds = pipe_out[0].as_cpu().as_array()  # as_cpu() to copy the data back to CPU memory
print(data_gds.shape)
plot_batch(data_gds)

