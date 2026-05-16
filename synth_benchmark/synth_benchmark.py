import subprocess
import argparse
import os
import pandas as pd
import itertools

load_type = {'SEQ_READ': 0, 'SEQ_WRITE':1, 'RAND_READ': 2, 'RAND_WRITE': 3}
transfer_type = {'GDS': 0, 'NONGDS': 2, 'PAGECACHE': 4}

# Usage [using config file]: gdsio rw-sample.gdsio
# Usage [using cmd line options]:./gdsio
#          -f <file name>
#          -D <directory name>
#          -d <gpu_index (refer nvidia-smi)>
#          -n <numa node>
#          -m <memory type(0 - (cudaMalloc), 1 - (cuMem), 2 - (cudaMallocHost), 3 - (malloc) 4 - (mmap))>
#          -w <number of threads for a job>
#          -s <file size(K|M|G)>
#          -o <start offset(K|M|G)>
#          -i <io_size(K|M|G)> <min_size:max_size:step_size>
#          -p <enable nvlinks>
#          -b <skip bufregister>
#          -o <start file offset>
#          -V <verify IO>
#          -x <xfer_type>
#          -I <(read) 0|(write)1| (randread) 2| (randwrite) 3>
#          -T <duration in seconds>
#          -k <random_seed> (number e.g. 3456) to be used with random read/write>
#          -U <use unaligned(4K) random offsets>
#          -R <fill io buffer with random data>
#          -F <refill io buffer with random data during each write>
#          -B

# xfer_type:
# 0 - Storage->GPU (GDS)
# 1 - Storage->CPU
# 2 - Storage->CPU->GPU
# 3 - Storage->CPU->GPU_ASYNC
# 4 - Storage->PAGE_CACHE->CPU->GPU
# 5 - Storage->GPU_ASYNC_STREAM
# 6 - Storage->GPU_BATCH
# 7 - Storage->GPU_BATCH_STREAM


def init_gds_files(gdsio_path, output_dir, file_size, device, workers):
    cmd = [gdsio_path, '-D', output_dir, '-d', device, '-T', '1', '-s', file_size, '-w', workers, '-I', 3]
    cmd = [str(x) for x in cmd]
    subprocess.run(cmd, check=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run synthetic GDS benchmarks with gdsio.")
    parser.add_argument("--gdsio-path", default="/usr/local/cuda/gds/tools/gdsio")
    parser.add_argument("--output-dir", default=os.environ.get("BEEOND", "/tmp"))
    parser.add_argument(
        "--results-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory for result CSV files.",
    )
    return parser.parse_args()

def main(gdsio_path, output_dir, results_dir):
    file_size = '30G'
    io_sweep = {
        'io': ['128K', '256K', '512K', '1M', '4M', '16M', '64M', '128M'], 
        'threads': [8], 
        'device': [0], 
        'numa_node': [0], 
        'load': ['RAND_READ', 'SEQ_READ'],
        'transfer': ['GDS', 'NONGDS'],
        'nvlinks': [False]
    } 
    thread_sweep = {
        'io': ['4k'],
        'threads': [1, 4, 8, 16, 32, 64, 128, 256, 512],
        'device': [0],
        'numa_node': [0],
        'load': ['RAND_READ', 'SEQ_READ'],
        'transfer': ['GDS', 'NONGDS'],
        'nvlinks': [False]
    }
    device_sweep = {
        'io': ['4K'],                 # keep IO size fixed for path sweep
        'threads': [8],               # keep threads fixed (adjust if needed)
        'device': [0, 1, 2, 3],       # GPU indices to sweep (nvidia-smi)
        'numa_node': [0, 1],          # NUMA nodes to sweep
        'nvlinks': [False, True],     # enable nvlinks (-p) toggle / dynamic routing
        'load': ['RAND_READ', 'SEQ_READ'],
        'transfer': ['GDS', 'NONGDS'],
    }
    
    time = '30'

    sweeps = {'io_sweep': io_sweep, 'thread_sweep': thread_sweep, 'device_sweep': device_sweep}
        
    for sweep_name, sweep_config in sweeps.items():
        print(f"--- Running {sweep_name} ---")
        res_dict = {'Transfer Type': [], 'Threads': [], 'Throughput (GiB/s)': [], 'Latency (usec)': [], 'IO Size': [], 'Device': [], 'NUMA': [], 'Load': [], 'NVLink': []}

        keys = list(sweep_config.keys())
        for combo in itertools.product(*sweep_config.values()):
            params = dict(zip(keys, combo))
            
            io_size, thread, dev, numa, use_nvlink, load, trans_name = (
                params['io'], params['threads'], params['device'], 
                params['numa_node'], params['nvlinks'], params['load'], params['transfer']
            )

            # Re-init files if needed for this max threads / device combo
            if not os.path.isfile(os.path.join(output_dir, f'gdsio.{thread - 1}')):
                init_gds_files(gdsio_path, output_dir, file_size, dev, thread)

            base_cmd = [gdsio_path, '-D', output_dir, '-T', time, '-s', file_size]
            new_cmd = base_cmd + ['-i', io_size, '-w', thread, '-x', transfer_type[trans_name], '-I', load_type[load], '-d', dev, '-n', numa]
            
            if use_nvlink:
                new_cmd.append('-p')

            new_cmd = [str(x) for x in new_cmd]
            print('Running', new_cmd)
            res = subprocess.run(new_cmd, check=True, capture_output=True, text=True)
            res_tokens = res.stdout.split()
            latency = float(res_tokens[res_tokens.index('Avg_Latency:') + 1])
            throughput = float(res_tokens[res_tokens.index('Throughput:') + 1])
            print('latency', latency, 'throughput', throughput)

            res_dict['Transfer Type'].append(trans_name)
            res_dict['Threads'].append(thread)
            res_dict['IO Size'].append(io_size)
            res_dict['Latency (usec)'].append(latency)
            res_dict['Throughput (GiB/s)'].append(throughput)
            res_dict['Device'].append(dev)
            res_dict['NUMA'].append(numa)
            res_dict['Load'].append(load)
            res_dict['NVLink'].append(use_nvlink)

        df = pd.DataFrame.from_dict(res_dict)
        os.makedirs(results_dir, exist_ok=True)
        df.to_csv(os.path.join(results_dir, f'{sweep_name}_results.csv'), index=False)


if __name__ == '__main__':
    args = parse_args()
    main(args.gdsio_path, args.output_dir, args.results_dir)

