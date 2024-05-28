import re
import numpy as np
import argparse
from itertools import chain
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime

dur_exp = r"[0-9][0-9]H [0-5][0-9]min [0-5][0-9]s"
log_dur_exp = r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"


def duration_between_dates(start_time, end_time):
    start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

    duration = (end_time - start_time).total_seconds()
    d_minutes = duration // 60
    return (d_minutes // 60, d_minutes % 60, duration % 60)


def get_duration(log_file):
    first_line, last_line = None, None
    with open(log_file, 'r') as f:
        for (i, line) in enumerate(f):
            if i == 0:
                first_line = line
            else:
                # keep updating last line
                last_line = line

            if 'Duration' in line:
                time_s = re.findall(dur_exp, line)
                assert len(time_s) == 1
                duration = list(map(int, re.findall(r'\d+', time_s[0])))
                assert len(duration) == 3
                return duration

    # if duration is not found in the log file,
    # compute it based on the first and last logs
    start_time = re.findall(log_dur_exp, first_line)[0]
    end_time = re.findall(log_dur_exp, last_line)[0]
    return duration_between_dates(start_time, end_time)


def duration_in_hour(durations):
    sum_dur = np.array(durations).sum(axis=0)
    return (sum_dur[0] + sum_dur[1]/60 + sum_dur[2]/3600)


if __name__ == "__main__":
    # argparser
    parser = argparse.ArgumentParser('duration')
    parser.add_argument('folders', nargs='*')
    args = parser.parse_args()

    log_files = list(
        chain(*[Path(folder).resolve().absolute().glob('**/logs.log')
                for folder in args.folders]))

    durations = [get_duration(log_file) for log_file in tqdm(log_files)]
    print(f"total duration: {duration_in_hour(durations):.2f} hours")
