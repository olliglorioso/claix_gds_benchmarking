#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_GDSIO_BIN = "/usr/local/cuda/gds/tools/gdsio"
DEFAULT_CASE_DIR = "gdsio_cases"
DEFAULT_LOG_DIR = "logs"
IOSTAT_COMMAND = ["iostat", "1"]
GDS_STAT_COMMANDS = (["gds_stats", "-l", "1"], ["gds_stat", "-l", "1"])


@dataclass(frozen=True)
class BenchmarkCase:
    suite: str
    path: Path

    @property
    def case_id(self) -> str:
        return self.path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run static gdsio benchmark config files with monitoring.",
    )
    parser.add_argument(
        "--suite",
        choices=["main", "topology"],
        help="Restrict execution to one named suite.",
    )
    parser.add_argument(
        "--case-dir",
        default=DEFAULT_CASE_DIR,
        help=f"Directory containing suite subdirectories of .gdsio files (default: {DEFAULT_CASE_DIR}).",
    )
    return parser.parse_args()


def discover_cases(case_dir: Path, suite: str | None) -> list[BenchmarkCase]:
    suites = [suite] if suite else ["main", "topology"]
    cases: list[BenchmarkCase] = []

    for suite_name in suites:
        suite_dir = case_dir / suite_name
        cases.extend(
            BenchmarkCase(suite_name, path)
            for path in sorted(suite_dir.glob("*.gdsio"))
            if path.is_file()
        )

    return cases


def command_exists(command: list[str]) -> bool:
    return shutil.which(command[0]) is not None


def resolve_gds_stat_command() -> list[str] | None:
    for command in GDS_STAT_COMMANDS:
        if command_exists(command):
            return command
    return None


def start_monitor(command: list[str] | None, log_path: Path) -> subprocess.Popen[str] | None:
    if command is None:
        print(f"WARN: gds_stat command not found, skipping {log_path.name}")
        log_path.touch()
        return None
    if not command_exists(command):
        print(f"WARN: command not found, skipping {' '.join(command)}")
        log_path.touch()
        return None

    handle = log_path.open("w")
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._log_handle = handle  # type: ignore[attr-defined]
    return process


def stop_monitor(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()

    handle = getattr(process, "_log_handle", None)
    if handle is not None:
        handle.close()


def run_case(
    case: BenchmarkCase,
    gdsio_bin: str,
    log_dir: Path,
    gds_stat_command: list[str] | None,
) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gdsio_log = log_dir / f"{case.case_id}.{timestamp}.gdsio.log"
    iostat_log = log_dir / f"{case.case_id}.{timestamp}.iostat.log"
    gds_stat_log = log_dir / f"{case.case_id}.{timestamp}.gds_stat.log"

    print("------------------------------------------------------------")
    print(f"START:         {case.case_id}")
    print(f"CONFIG:        {case.path}")
    print(f"GDSIO LOG:     {gdsio_log}")
    print(f"IOSTAT LOG:    {iostat_log}")
    print(f"GDS_STAT LOG:  {gds_stat_log}")

    iostat_process = start_monitor(IOSTAT_COMMAND, iostat_log)
    gds_stat_process = start_monitor(gds_stat_command, gds_stat_log)

    try:
        with gdsio_log.open("w") as log_file:
            process = subprocess.Popen(
                [gdsio_bin, str(case.path)],
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
        stop_monitor(iostat_process)
        stop_monitor(gds_stat_process)

    if success:
        print(f"OK:            {case.case_id}")
    else:
        print(f"FAIL:          {case.case_id}")
    print()
    return success


def main() -> int:
    args = parse_args()
    case_dir = Path(args.case_dir)
    log_dir = Path(DEFAULT_LOG_DIR)

    if not case_dir.is_dir():
        print(f"ERROR: missing case directory: {case_dir}", file=sys.stderr)
        return 1

    if not os.access(DEFAULT_GDSIO_BIN, os.X_OK):
        print(f"ERROR: gdsio binary not executable: {DEFAULT_GDSIO_BIN}", file=sys.stderr)
        return 1

    selected = discover_cases(case_dir, args.suite)
    if not selected:
        print("No .gdsio cases matched the requested suite.", file=sys.stderr)
        return 1

    log_dir.mkdir(parents=True, exist_ok=True)
    gds_stat_command = resolve_gds_stat_command()
    if gds_stat_command is None:
        print("WARN: neither gds_stats nor gds_stat was found; only gdsio and iostat will run.")

    failures = 0
    for case in selected:
        if not run_case(case, DEFAULT_GDSIO_BIN, log_dir, gds_stat_command):
            failures += 1

    print(
        "RAW LOGS: "
        f"{log_dir}/ contains per-case .gdsio.log, .iostat.log, and .gds_stat.log files."
    )
    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1

    print(f"All {len(selected)} case(s) completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
