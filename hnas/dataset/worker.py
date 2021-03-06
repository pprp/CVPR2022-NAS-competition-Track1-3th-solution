import os
import six
import sys
import paddle
import numpy as np
import traceback

from paddle.fluid import core
from paddle.fluid.dataloader.fetcher import _IterableDatasetFetcher, _MapDatasetFetcher
from paddle.fluid.multiprocess_utils import _cleanup_mmap, CleanupFuncRegistrar, MP_STATUS_CHECK_INTERVAL
from paddle.fluid.dataloader.flat import _flatten_batch

import queue
from .random_size_crop import MyRandomResizedCrop


class _IterableDatasetStopIteration(object):
    def __init__(self, worker_id):
        self.worker_id = worker_id


class _DatasetKind(object):
    MAP = 0
    ITER = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collate_batch, collate_fn,
                       drop_last):
        if kind == _DatasetKind.MAP:
            return _MapDatasetFetcher(dataset, auto_collate_batch, collate_fn,
                                      drop_last)
        elif kind == _DatasetKind.ITER:
            return _IterableDatasetFetcher(dataset, auto_collate_batch,
                                           collate_fn, drop_last)
        else:
            raise NotImplementedError("unknown Dataset kind {}".format(kind))


class ParentWatchDog(object):
    def __init__(self):
        self._parent_pid = os.getppid()
        self._parent_alive = True

    def is_alive(self):
        if self._parent_alive:
            self._parent_alive = os.getppid() == self._parent_pid
        return self._parent_alive


# worker information for each workers, used for splitting data copy
# for IteratorDataset in worker processes.
_worker_info = None


class WorkerInfo(object):
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError("Cannot assign attributes to {} objects".format(
                self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)


class _WorkerException(object):
    def __init__(self, worker_id, exc_info=None):
        self.worker_id = worker_id
        exc_info = exc_info or sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def reraise(self):
        msg = "DataLoader worker({}) caught {} with message:\n{}".format(
            self.worker_id, self.exc_type.__name__, self.exc_msg)
        if getattr(self.exc_type, "message", None):
            raise self.exc_type(message=msg)
        raise self.exc_type(msg)


def _worker_loop(dataset, dataset_kind, indices_queue, out_queue, done_event,
                 auto_collate_batch, collate_fn, init_fn, worker_id,
                 num_workers, use_shared_memory):
    try:
        # NOTE: [ mmap files clear ] When the child process exits unexpectedly,
        # some shared memory objects may have been applied for but have not yet
        # been put into the inter-process Queue. This part of the object needs
        # to be cleaned up when the process ends.
        CleanupFuncRegistrar.register(_cleanup_mmap)

        # set signal handler
        core._set_process_signal_handler()

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, dataset=dataset)

        init_exception = None
        try:
            if init_fn is not None:
                init_fn(worker_id)
            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collate_batch, collate_fn, True)
        except:
            init_exception = _WorkerException(worker_id)

        iterator_drained = False
        parent_watch_dog = ParentWatchDog()

        while parent_watch_dog.is_alive():
            try:
                data = indices_queue.get(MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            # None as poison piil, so worker event should be set
            if data is None:
                assert done_event.is_set() or iterator_drained, \
                        "get None when worker done_event set"
                break
            # If worker done event is set but get still get data in
            # indices_queue, remaining data should be get and skipped.
            if done_event.is_set() or iterator_drained:
                continue

            idx, indices = data
            """ Added """
            MyRandomResizedCrop.sample_image_size(idx)
            """ Added """
            try:
                if init_exception is not None:
                    batch = init_exception
                    init_exception = None
                else:
                    # NOTE: GPU tensor operation is not supported in sub-process
                    #       but default device is GPU in paddle-gpu version, which
                    #       may copy CPU tensor to GPU even if users want to use
                    #       CPU tensor operation, so we add CPUPlace guard here
                    #       to make sure tensor will be operated only on CPU
                    with paddle.fluid.dygraph.guard(place=paddle.CPUPlace()):
                        batch = fetcher.fetch(indices)
            except Exception as e:
                if isinstance(
                        e, StopIteration) and dataset_kind == _DatasetKind.ITER:
                    out_queue.put(_IterableDatasetStopIteration(worker_id))
                    iterator_drained = True
                else:
                    out_queue.put((idx, _WorkerException(worker_id), None))
            else:
                if isinstance(batch, _WorkerException):
                    out_queue.put((idx, batch, None))
                batch, structure = _flatten_batch(batch)
                if use_shared_memory:
                    tensor_list = [
                        core._array_to_share_memory_tensor(b)
                        if isinstance(b, np.ndarray) else b._share_memory()
                        for b in batch
                    ]
                    out_queue.put((idx, tensor_list, structure))
                    core._remove_tensor_list_mmap_fds(tensor_list)
                else:
                    out_queue.put((idx, batch, structure))
    except KeyboardInterrupt:
        # NOTE: Main process will raise KeyboardInterrupt anyways, ignore it in child process
        pass
    except:
        six.reraise(*sys.exc_info())
    finally:
        if use_shared_memory:
            _cleanup_mmap()