import h5py
import numpy as np
import socket
import os
import csv

from scipy import interpolate
from timeit import default_timer


def write_results(data, results_file):
    hostname = socket.gethostname()
    if not os.path.isfile(results_file):
        write_header = True
    else:
        write_header = False

    data['hostname'] = hostname
    fieldnames = list(data.keys())
    with open(results_file, 'a') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(data)


def to_hdf5(data, filename, datakey='data', additional=None):
    with h5py.File(filename, 'w') as f:
        f.create_dataset(datakey, data=data, dtype=data.dtype)
        if additional is not None:
            for k, v in additional.items():
                f.create_dataset(k, data=v, dtype=v.dtype)


class Timer(object):
    def __init__(self, profiler, section, action):
        self.timer = default_timer
        self.profiler = profiler
        self.section = section
        self.action = action

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.profiler.increment(self.section, self.action, self.elapsed)


class Profiler(object):
    def __init__(self):
        self.timings = {}
        self.counts = {}

    def get_timer(self, section, action):
        return Timer(self, section, action)

    def increment(self, section, action, elapsed):
        # Warning: Not thread safe
        section_timings = self.timings.get(section, {})
        section_timings[action] = section_timings.get(action, 0) + elapsed
        self.timings[section] = section_timings

        section_counts = self.counts.get(section, {})
        section_counts[action] = section_counts.get(action, 0) + 1
        self.counts[section] = section_counts

    def summary(self):
        summary = '****************'
        for section, section_timings in self.timings.items():
            summary += '\nIn section %s:' % section
            for action, action_time in section_timings.items():
                summary += '\n\tAction %s: %f (%d)' \
                           % (action, action_time,
                              self.counts[section][action])
        summary += '\n****************'
        return summary

    def get_dict(self):
        results = {}
        for s_n, s_dict in self.timings.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_timing' % (s_n, a_n)] = a_time

        for s_n, s_dict in self.counts.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_counts' % (s_n, a_n)] = a_time

        return results


def mat2vec(mat):
    return np.ravel(mat)


def vec2mat(vec, shape):
    if vec.shape == shape:
        return vec
    return np.reshape(vec, shape)


def clip_boundary_and_numpy(mat, nbl):
    return np.array(mat.data[:])[nbl:-nbl, nbl:-nbl]


def reinterpolate(shot, new_nt, old_dt, order=3):
    old_nt, ntraces = shot.shape

    if old_nt == new_nt:
        return shot

    old_tn = float(old_nt*old_dt)

    new_dt = old_tn/new_nt

    new_shot = np.zeros((new_nt, ntraces))

    oldt = np.array([i*new_dt for i in range(old_nt)])
    newt = np.array([i*new_dt for i in range(new_nt)])

    for i in range(ntraces):
        tck = interpolate.splrep(oldt, shot[:, i], s=0, k=order)
        new_shot[:, i] = interpolate.splev(newt, tck)

    return new_shot

def m_to_vp(m):
    return np.sqrt(1./m)

def vp_to_m(vp):
    return 1./vp**2
