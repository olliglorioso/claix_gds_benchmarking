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

data_dir = os.path.join(data_dir_2d, 'SER00001')

@pipeline_def(batch_size=batch_size, num_threads=3, device_id=0)
def pipe1():
    data = fn.readers.numpy(device='cpu', file_root=data_dir, file_filter='*.npy')
    return data

def run(p):
    p.build()  # build the pipeline
    outputs = p.run()  # Run once
    # Getting the batch as a list of numpy arrays, for displaying
    batch = [np.array(outputs[0][s]) for s in range(batch_size)]
    return batch

data1 = run(pipe1())
print(data1)
