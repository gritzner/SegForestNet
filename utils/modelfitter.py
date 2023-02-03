import core
import os
import time
import datetime
import numpy as np
import json
import concurrent.futures
import queue
import numbers
import traceback
import types
import torch


class Queue(queue.Queue):
    def put(self, item):
        try:
            super().put(item, timeout=2)
            return False
        except queue.Full:
            return True

    def get(self):
        try:
            return super().get(timeout=2)
        except queue.Empty:
            return None


class ModelFitter():
    def __init__(self, config):
        self.config = config
        self.output_float_precision = getattr(config, "output_float_precision", 4)
        self.output_set = set()
        self.history_filename = getattr(config, "history_filename", "history.json") 
        self.history = {}
        max_queue_size = getattr(config, "max_queue_size", 4)
        self.max_queue_size = (max_queue_size - 1) if max_queue_size > 1 else 1
        
    def pre_epoch(self, epoch):
        pass
    
    def pre_evaluate(self, epoch):
        pass
    
    def pre_train(self, epoch, batch, batch_data):
        pass
    
    def train(self, epoch, batch, batch_data, metrics):
        pass
    
    def post_train(self, epoch, batch, batch_data, metrics):
        pass
    
    def post_evaluate(self, epoch):
        pass
    
    def post_epoch(self, epoch, metrics):
        pass
    
    def finalize(self):
        pass

    def run(self):
        epochs = self.config.terminate_early if getattr(self.config,"terminate_early",-1)>0 else self.config.epochs
        for epoch in range(epochs):
            epoch_timestamp = time.perf_counter()
            print(f"starting epoch {epoch+1} at", datetime.datetime.now().time().strftime("%H:%M:%S"))
            self.pre_epoch(epoch)
            self._fit_epoch(epoch)
            metrics = {key: np.mean(self.history[key][-self.num_mini_batches:]) if isinstance(self.history[key][-1], numbers.Number) and not isinstance(self.history[key][-1], bool) and "loss" in key else self.history[key][-1] for key in self._metrics_keys}
            metrics = types.SimpleNamespace(**metrics)
            self.post_epoch(epoch, metrics)
            metrics = metrics.__dict__
            for key in self._metrics_keys:
                del metrics[key]
            for key, value in metrics.items():
                if not key in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            if core.output_path != None and self.history_filename != None:
                filename = f"{core.output_path}/{self.history_filename}"
                print(f"saving history to '{filename}'...")
                with open(filename, "w") as f:
                    json.dump(self.history, f)
            epoch_timestamp = time.perf_counter() - epoch_timestamp
            self._progress(epoch, -1, epoch_timestamp, metrics)
        self.finalize()
        
    def evaluate(self, epoch, **params):
        num_train_mini_batches = self.num_mini_batches
        self.num_mini_batches = 0
        torch.set_grad_enabled(False)
        self.eval_params = types.SimpleNamespace(enabled=True, **params)
        self.pre_evaluate(epoch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            try:
                self._terminate_workers = False
                q_train = Queue(maxsize=self.max_queue_size)
                executor.submit(self._get_batches, epoch, q_train)
                q_metrics = Queue(maxsize=self.max_queue_size)
                executor.submit(self._train, epoch, q_train, q_metrics)
                for batch in range(self.num_mini_batches):
                    data = q_metrics.get()
                    while not data:
                        if self._terminate_workers:
                            raise RuntimeError("terminating main thread")
                        data = q_metrics.get()
                    data, metrics = data
                    self.post_train(epoch, batch, data, metrics)
            except:
                traceback.print_exc()
                self._terminate_workers = True
        if self._terminate_workers:
            raise RuntimeError("terminated")
        self.post_evaluate(epoch)
        torch.set_grad_enabled(True)
        self.num_mini_batches = num_train_mini_batches
        
    def _fit_epoch(self, epoch):
        self._last_line_length = None
        self._metrics_keys = set()
        self.eval_params = types.SimpleNamespace(enabled=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            try:
                self._terminate_workers = False
                q_train = Queue(maxsize=self.max_queue_size)
                executor.submit(self._get_batches, epoch, q_train)
                q_metrics = Queue(maxsize=self.max_queue_size)
                executor.submit(self._train, epoch, q_train, q_metrics)
                batch_timestamp = time.perf_counter()
                for batch in range(self.num_mini_batches):
                    data = q_metrics.get()
                    while not data:
                        if self._terminate_workers:
                            raise RuntimeError("terminating main thread")
                        data = q_metrics.get()
                    data, metrics = data
                    self.post_train(epoch, batch, data, metrics)
                    metrics = metrics.__dict__
                    for key, value in metrics.items():
                        if not key in self._metrics_keys:                            
                            self._metrics_keys.add(key)
                        if not key in self.history:
                            self.history[key] = []
                        values = self.history[key]
                        values.append(value)
                        if isinstance(value, numbers.Number) and not isinstance(value, bool):
                            if "loss" in key:
                                metrics[key] = np.mean(values[-(batch+1):])
                    self._progress(epoch, batch, (time.perf_counter() - batch_timestamp) / (batch + 1), metrics)
            except:
                traceback.print_exc()
                self._terminate_workers = True
        if self._terminate_workers:
            raise RuntimeError("terminated")
        print("")
                                
    def _get_batches(self, epoch, q):
        if self.eval_params.enabled:
            torch.set_grad_enabled(False)
        try:
            for batch in range(self.num_mini_batches):
                data = types.SimpleNamespace()
                self.pre_train(epoch, batch, data)
                while q.put(data):
                    if self._terminate_workers:
                        raise RuntimeError("terminating first worker")
        except:
            traceback.print_exc()
            self._terminate_workers = True
        if self.eval_params.enabled:
            torch.set_grad_enabled(True)
            
    def _train(self, epoch, q_in, q_out):
        if self.eval_params.enabled:
            torch.set_grad_enabled(False)
        try:
            for batch in range(self.num_mini_batches):
                data = q_in.get()
                while not data:
                    if self._terminate_workers:
                        raise RuntimeError("terminating second worker during get()")
                    data = q_in.get()
                metrics = types.SimpleNamespace()
                self.train(epoch, batch, data, metrics)
                while q_out.put((data, metrics)):
                    if self._terminate_workers:
                        raise RuntimeError("terminating second worker during put()")
        except:
            traceback.print_exc()           
            self._terminate_workers = True
        if self.eval_params.enabled:
            torch.set_grad_enabled(True)

    def _progress(self, epoch, batch, elapsed_time, metrics):
        epochs = self.config.terminate_early if getattr(self.config,"terminate_early",-1)>0 else self.config.epochs
        s = f"epoch = {epoch+1}/{epochs}"
        if batch >= 0:
            s = f"{s}, iteration = {batch+1}/{self.num_mini_batches}"
        if elapsed_time < 1:            
            elapsed_time *= 1000
            unit = "ms"
            if elapsed_time < 1:
                elapsed_time *= 1000
                unit = "us"
            elapsed_time = round(elapsed_time)
            s = f"{s}, time = {elapsed_time}{unit}"
        elif elapsed_time < 60:
            s = f"{s}, time = {elapsed_time:.2f}s"
        else:
            elapsed_time /= 60
            unit = "min"
            if elapsed_time >= 60:
                elapsed_time /= 60
                unit = "h"
                if elapsed_time >= 24:
                    elapsed_time /= 24
                    unit = "d"
            s = f"{s}, time = {elapsed_time:.1f}{unit}"
        for key, value in metrics.items():
            if len(self.output_set) > 0 and key not in self.output_set:
                continue
            t = type(value)
            if t == float or t == np.float32 or t == np.float64:
                format_s = f"%s, %s = %.{self.output_float_precision}f"
                s = format_s%(s, key, value)
            else:
                s = f"{s}, {key} = {value}"
        if self._last_line_length and batch >= 0 and len(s) < self._last_line_length:
            s += (self._last_line_length-len(s))*" "
        print(s, end="\r" if batch >= 0 else "\n\n")
        self._last_line_length = len(s)
