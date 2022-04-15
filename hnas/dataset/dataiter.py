import os
import six
import sys
import time
import signal
import numbers
import logging
import itertools
import threading
import numpy as np
import multiprocessing
from collections import namedtuple
from paddle.fluid.framework import _set_expected_place, _current_expected_place

import queue

import paddle.fluid.core as core
import paddle.fluid as fluid

from paddle.fluid.dataloader import dataloader_iter as dli
from paddle.io import DataLoader
from .worker import _worker_loop


def init_workers(self):
    # multiprocess worker and indice queue list initial as empty
    self._workers = []
    self._worker_status = []
    self._indices_queues = []
    self._workers_idx_cycle = itertools.cycle(range(self._num_workers))

    # create data_queue for workers
    self._data_queue = multiprocessing.Queue()

    # event for workers and thread, thread event is only need 
    # in multi-processing mode
    self._workers_done_event = multiprocessing.Event()
    self._thread_done_event = threading.Event()

    for i in range(self._num_workers):
        indices_queue = multiprocessing.Queue()
        self._indices_queues.append(indices_queue)
        worker = multiprocessing.Process(
            target=_worker_loop,
            args=(self._dataset, self._dataset_kind, indices_queue,
                    self._data_queue, self._workers_done_event,
                    self._auto_collate_batch, self._collate_fn,
                    self._worker_init_fn, i, self._num_workers,
                    self._use_shared_memory))
        worker.daemon = True
        worker.start()
        self._workers.append(worker)
        self._worker_status.append(True)

    core._set_process_pids(id(self), tuple(w.pid for w in self._workers))
    fluid.multiprocess_utils._set_SIGCHLD_handler()

dli._DataLoaderIterMultiProcess._init_workers = init_workers

def iter(self):
    if self.num_workers == 0:
        raise NotImplementedError
    else:
        return dli._DataLoaderIterMultiProcess(self)

DataLoader.__iter__ = iter
