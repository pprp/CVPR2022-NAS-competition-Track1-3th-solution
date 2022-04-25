import os
import six
import pickle
import numbers

import paddle
from paddle import callbacks


class LRSchedulerM(callbacks.LRScheduler):
    def __init__(self, by_step=False, by_epoch=True, warm_up=True):
        super().__init__(by_step, by_epoch)
        assert by_step ^ warm_up
        self.warm_up = warm_up

    def on_epoch_end(self, epoch, logs=None):
        if self.by_epoch and not self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()

    def on_train_batch_end(self, step, logs=None):
        if self.by_step or self.warm_up:
            if self.model._optimizer and hasattr(
                self.model._optimizer, '_learning_rate') and isinstance(
                    self.model._optimizer._learning_rate, paddle.optimizer.lr.LRScheduler):
                self.model._optimizer._learning_rate.step()
            if self.model._optimizer._learning_rate.last_epoch >= self.model._optimizer._learning_rate.warmup_steps:
                self.warm_up = False


class EvalCheckpoint(callbacks.Callback):
    def __init__(self, model_path=None):
        super().__init__()
        self.model_path = model_path
        self.eval_flag = True

    def on_eval_begin(self, logs=None):
        if self.model_path is not None and self.eval_flag:
            print('Eval: load checkpoint at {}'.format(self.model_path))
            self.model.load(self.model_path, reset_optimizer=False)
            self.eval_flag = False


class MyModelCheckpoint(callbacks.ModelCheckpoint):
    def __init__(self, save_freq=1, save_dir=None, resume=None, phase=None):
        super().__init__(save_freq, save_dir)
        self.resume = resume
        self.phase = phase

    def load_state_from_path(self, path):
        if not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            return pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        
    def on_train_begin(self, logs=None):
        # if self.phase is not None:
        #     path = '{}/final'.format(self.phase)
        #     print('Phase: load checkpoint at {}'.format(os.path.abspath(path)))
        #     self.model.load(path, reset_optimizer=True)

        if self.resume is not None:
            path = '{}/final'.format(self.resume)
            print('Resume: load checkpoint at {}'.format(os.path.abspath(path)))
            opt_path = path + ".pdopt"
            optim_state = self.load_state_from_path(opt_path)
            self.model.start_epoch = optim_state['epoch'] + 1
            print('start epoch: ', self.model.start_epoch)
            self.model.load(path)

    def on_epoch_end(self, epoch, logs=None):
        if self._is_save() and self.epoch % self.save_freq == 0:
            path = '{}/{}'.format(self.save_dir, epoch)
            print('MY: save checkpoint at {}'.format(os.path.abspath(path)))
            self.model.save(path)
        path = '{}/final'.format(self.save_dir)
        print('MY: save checkpoint at {}'.format(os.path.abspath(path)))
        self.model.save(path, training=True)


class MyModelCheckpoint2(callbacks.ModelCheckpoint):
    def __init__(self, save_freq=1, save_dir=None, resume=None, phase=None):
        super().__init__(save_freq, save_dir)
        self.resume = resume
        self.phase = phase

    def load_state_from_path(self, path):
        if not os.path.exists(path):
            return
        with open(path, 'rb') as f:
            return pickle.load(f) if six.PY2 else pickle.load(f, encoding='latin1')
        
    def on_train_begin(self, logs=None):
        if self.resume is not None:
            path = '{}/80'.format(self.resume)
            self.model.start_epoch = 81
            print('start epoch: ', self.model.start_epoch)
            self.model.load(path)

    def on_epoch_end(self, epoch, logs=None):
        if self._is_save() and self.epoch % self.save_freq == 0:
            path = '{}/{}'.format(self.save_dir, epoch)
            print('MY: save checkpoint at {}'.format(os.path.abspath(path)))
            self.model.save(path)
        path = '{}/final'.format(self.save_dir)
        print('MY: save checkpoint at {}'.format(os.path.abspath(path)))
        self.model.save(path, training=True)


class TensorboardX(callbacks.VisualDL):
    def __init__(self, log_dir=None):
        super().__init__(log_dir)

    def _updates(self, logs, mode):
        if not self._is_write():
            return
        if not hasattr(self, 'writer'):
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)

        metrics = getattr(self, '%s_metrics' % (mode))
        current_step = getattr(self, '%s_step' % (mode))

        if mode == 'train':
            total_step = current_step
        else:
            total_step = self.epoch
        for k in metrics:
            if k in logs and total_step % 100 == 0 and mode == 'train':
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    for idx, temp_value in enumerate(logs[k], 1):
                        self.writer.add_scalar(tag=f'{temp_tag}_{idx}', global_step=total_step, scalar_value=temp_value)
                elif isinstance(logs[k], numbers.Number):
                    temp_value = logs[k]
                    self.writer.add_scalar(tag=temp_tag, global_step=total_step, scalar_value=temp_value)
                else:
                    continue
            if k in logs and mode == 'eval':
                temp_tag = mode + '/' + k

                if isinstance(logs[k], (list, tuple)):
                    for idx, temp_value in enumerate(logs[k], 1):
                        self.writer.add_scalar(tag=f'{temp_tag}_{idx}', global_step=total_step, scalar_value=temp_value)
                elif isinstance(logs[k], numbers.Number):
                    temp_value = logs[k]
                    self.writer.add_scalar(tag=temp_tag, global_step=total_step, scalar_value=temp_value)
                else:
                    continue
                
    def on_train_end(self, logs=None):
        if hasattr(self, 'writer'):
            self.writer.close()
            delattr(self, 'writer')

    def on_eval_end(self, logs=None):
        if self._is_write():
            self._updates(logs, 'eval')

            if (not hasattr(self, '_is_fit')) and hasattr(self, 'writer'):
                self.writer.close()
                delattr(self, 'writer')
                
    def on_train_batch_end(self, step, logs=None):
        logs = logs or {}
        logs['lr'] = self.model._optimizer.get_lr()
        self.train_step += 1
        if self._is_write():
            self._updates(logs, 'train')

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        assert self.epochs
        self.train_metrics = self.params['metrics'] + ['lr']
        assert self.train_metrics
        self._is_fit = True
        self.train_step = 0