#!/usr/bin/env python3
"""Run a command, sampling /proc/PID/status VmHWM for peak RSS.

Prints PEAK_RSS_KB:<n> at the end. Exit code mirrors the child.
Usage: run_with_peak_rss.py <logfile> <cmd...>
"""
import subprocess
import sys
import time


def vmhwm(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return None
    return None


def main():
    logfile, cmd = sys.argv[1], sys.argv[2:]
    peak = 0
    with open(logfile, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            v = vmhwm(proc.pid)
            if v is not None:
                peak = max(peak, v)
            time.sleep(1.0)
        v = vmhwm(proc.pid)
        if v is not None:
            peak = max(peak, v)
    print(f"PEAK_RSS_KB:{peak}")
    print(f"CHILD_EXIT:{proc.returncode}")
    sys.exit(0)


if __name__ == "__main__":
    main()
