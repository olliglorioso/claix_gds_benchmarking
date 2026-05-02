#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_GDSIO_BIN = "/usr/local/cuda/gds/tools/gdsio"
DEFAULT_LOG_DIR = "logs"
DEFAULT_RUNTIME = "60"
DEFAULT_DO_VERIFY = "0"
DEFAULT_TEMPLATE_FILE = "gdsio_template.gdsio"

@dataclass
class BenchmarkCase:
    suite: str
    rw: str
    transport: str
    bs: str
    size: str
    gpu: str

    @property
    def case_id(self) -> str:
        return (
            f"{self.suite}.{self.rw}.{self.transport}."
            f"bs{self.bs.lower()}.size{self.size.lower()}.gpu{self.gpu.lower()}"
        )

    @property
    def xfer_type(self) -> str:
        return "0" if self.transport == "gds" else "1"

    @property
    def name(self) -> str:
        return (
            f"beeond_h100_{self.suite}_{self.rw}_{self.transport}_"
            f"bs{self.bs.lower()}_size{self.size.lower()}_gpu{self.gpu.lower()}"
        )


CASES = [
    BenchmarkCase("main", "seqread", "gds", "4K", "4G", "all"),
    BenchmarkCase("main", "seqread", "gds", "4K", "32G", "all"),
    BenchmarkCase("main", "seqread", "gds", "1M", "4G", "all"),
    BenchmarkCase("main", "seqread", "gds", "1M", "32G", "all"),
    BenchmarkCase("main", "seqread", "gds", "4M", "4G", "all"),
    BenchmarkCase("main", "seqread", "gds", "4M", "32G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "4K", "4G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "4K", "32G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "1M", "4G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "1M", "32G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "4M", "4G", "all"),
    BenchmarkCase("main", "seqread", "nogds", "4M", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "4K", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "4K", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "1M", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "1M", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "4M", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "gds", "4M", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "4K", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "4K", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "1M", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "1M", "32G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "4M", "4G", "all"),
    BenchmarkCase("main", "seqwrite", "nogds", "4M", "32G", "all"),
    BenchmarkCase("main", "randread", "gds", "4K", "4G", "all"),
    BenchmarkCase("main", "randread", "gds", "4K", "32G", "all"),
    BenchmarkCase("main", "randread", "gds", "1M", "4G", "all"),
    BenchmarkCase("main", "randread", "gds", "1M", "32G", "all"),
    BenchmarkCase("main", "randread", "gds", "4M", "4G", "all"),
    BenchmarkCase("main", "randread", "gds", "4M", "32G", "all"),
    BenchmarkCase("main", "randread", "nogds", "4K", "4G", "all"),
    BenchmarkCase("main", "randread", "nogds", "4K", "32G", "all"),
    BenchmarkCase("main", "randread", "nogds", "1M", "4G", "all"),
    BenchmarkCase("main", "randread", "nogds", "1M", "32G", "all"),
    BenchmarkCase("main", "randread", "nogds", "4M", "4G", "all"),
    BenchmarkCase("main", "randread", "nogds", "4M", "32G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "4K", "4G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "4K", "32G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "1M", "4G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "1M", "32G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "4M", "4G", "all"),
    BenchmarkCase("main", "randwrite", "gds", "4M", "32G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "4K", "4G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "4K", "32G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "1M", "4G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "1M", "32G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "4M", "4G", "all"),
    BenchmarkCase("main", "randwrite", "nogds", "4M", "32G", "all"),
    BenchmarkCase("topology", "seqread", "gds", "4K", "32G", "0"),
    BenchmarkCase("topology", "seqread", "gds", "4K", "32G", "1"),
    BenchmarkCase("topology", "seqread", "gds", "4K", "32G", "2"),
    BenchmarkCase("topology", "seqread", "gds", "4K", "32G", "3"),
    BenchmarkCase("topology", "seqread", "gds", "1M", "32G", "0"),
    BenchmarkCase("topology", "seqread", "gds", "1M", "32G", "1"),
    BenchmarkCase("topology", "seqread", "gds", "1M", "32G", "2"),
    BenchmarkCase("topology", "seqread", "gds", "1M", "32G", "3"),
    BenchmarkCase("topology", "randread", "gds", "4K", "32G", "0"),
    BenchmarkCase("topology", "randread", "gds", "4K", "32G", "1"),
    BenchmarkCase("topology", "randread", "gds", "4K", "32G", "2"),
    BenchmarkCase("topology", "randread", "gds", "4K", "32G", "3"),
    BenchmarkCase("topology", "randread", "gds", "1M", "32G", "0"),
    BenchmarkCase("topology", "randread", "gds", "1M", "32G", "1"),
    BenchmarkCase("topology", "randread", "gds", "1M", "32G", "2"),
    BenchmarkCase("topology", "randread", "gds", "1M", "32G", "3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run gdsio cases from a built-in benchmark matrix."
    )
    parser.add_argument(
        "--suite",
        choices=["main", "topology"],
        help="Restrict execution to one named suite.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching case ids without running them.",
    )
    return parser.parse_args()


def render_jobs(case: BenchmarkCase) -> str:
    if case.gpu == "all":
        gpu_ids = ["0", "1", "2", "3"]
    else:
        gpu_ids = [case.gpu]

    job_blocks: list[str] = []
    for index, gpu_id in enumerate(gpu_ids, start=1):
        job_blocks.append(
            "\n".join(
                [
                    f"[job{index}]",
                    f"# H100 GPU {gpu_id}",
                    "numa_node=0",
                    f"gpu_dev_id={gpu_id}",
                    "num_threads=8",
                    f"directory=$BEEOND/gds_dir/gpu{gpu_id}",
                ]
            )
        )
    return "\n\n".join(job_blocks)


def render_config(
    template_text: str,
    case: BenchmarkCase,
    runtime: str,
    do_verify: str,
) -> str:
    return template_text.format(
        name=case.name,
        xfer_type=case.xfer_type,
        rw=case.rw,
        bs=case.bs,
        size=case.size,
        runtime=runtime,
        do_verify=do_verify,
        jobs=render_jobs(case),
    )


def run_case(
    case: BenchmarkCase,
    template_text: str,
    gdsio_bin: str,
    log_dir: Path,
    runtime: str,
    do_verify: str,
) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{case.case_id}.{timestamp}.log"
    config_text = render_config(template_text, case, runtime, do_verify)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".gdsio",
        prefix=f"{case.case_id}.",
        delete=False,
    ) as handle:
        handle.write(config_text)
        temp_path = Path(handle.name)

    print("------------------------------------------------------------")
    print(f"START: {case.case_id}")
    print(f"LOG:   {log_path}")

    try:
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                [gdsio_bin, str(temp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)

            success = process.wait() == 0
    finally:
        temp_path.unlink(missing_ok=True)

    if success:
        print(f"OK:    {case.case_id}")
    else:
        print(f"FAIL:  {case.case_id}")
    print()
    return success


def main() -> int:
    args = parse_args()
    log_dir = Path(DEFAULT_LOG_DIR)
    template_path = Path(DEFAULT_TEMPLATE_FILE)

    if not template_path.is_file():
        print(f"ERROR: missing template file: {template_path}", file=sys.stderr)
        return 1

    if not args.list and not os.access(DEFAULT_GDSIO_BIN, os.X_OK):
        print(f"ERROR: gdsio binary not executable: {DEFAULT_GDSIO_BIN}", file=sys.stderr)
        return 1

    selected = [case for case in CASES if args.suite is None or case.suite == args.suite]
    if not selected:
        print("No cases matched the provided filters.", file=sys.stderr)
        return 1

    if args.list:
        for case in selected:
            print(case.case_id)
        print(f"Listed {len(selected)} case(s).")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    template_text = template_path.read_text()

    failures = 0
    for case in selected:
        if not run_case(
            case,
            template_text,
            DEFAULT_GDSIO_BIN,
            log_dir,
            DEFAULT_RUNTIME,
            DEFAULT_DO_VERIFY,
        ):
            failures += 1

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1

    print(f"All {len(selected)} case(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
