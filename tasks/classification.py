import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import time


class Classification(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        
        self.config.smoothing = getattr(self.config, "smoothing", 0.6)

        self.dataset = core.create_object(datasets, config.dataset)
        self.num_mini_batches = int(np.ceil(self.dataset.training.x.shape[0] / self.config.mini_batch_size))
        
        shape = list(self.dataset.training.x.shape[-3:])
        self.model = core.create_object(models.classification, config.model, input_shape=shape, num_classes=self.dataset.num_classes)

        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
        
        print() # add a new line for nicer output formatting
        
        self.loss_func = nn.CrossEntropyLoss(weight=self.dataset.class_weights, ignore_index=self.dataset.ignore_class)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate.max_value)
        self.lr_scheduler = utils.LearningRateScheduler(
            self.optim, self.config.learning_rate.min_value,
            self.config.learning_rate.num_cycles, self.config.learning_rate.cycle_length_factor,
            self.num_mini_batches * self.config.epochs
        )
        
        self.conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        self.output_set.update(("loss", "acc", "val_loss", "val_acc"))
        self.shuffle_rng = np.random.RandomState(core.random_seeds[self.config.shuffle_seed])
        
    def pre_epoch(self, epoch):
        self.model.train()
        self.shuffle_map = self.shuffle_rng.permutation(self.dataset.training.x.shape[0])
        self.conf_mat.reset()
        
    def pre_train(self, epoch, batch, batch_data):
        indices = self.shuffle_map[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        batch_data.x = self.dataset.training.x[indices]
        batch_data.yt = self.dataset.training.y[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        self.optim.zero_grad()
        x = torch.from_numpy(batch_data.x).float().requires_grad_().to(core.device)
        yp = self.model(x)
        yt = torch.from_numpy(batch_data.yt).long().to(core.device)
        loss = self.loss_func(yp, yt)
        loss.backward()
        self.optim.step()
        metrics.learning_rate = self.lr_scheduler.get_lr()
        self.lr_scheduler.step(epoch * self.num_mini_batches + batch)
        metrics.loss = loss.item()
        batch_data.yp = yp.argmax(1).cpu().numpy()
        
    def post_train(self, epoch, batch, batch_data, metrics):
        self.conf_mat.add(batch_data.yt, batch_data.yp)
        current_metrics = self.conf_mat.compute_metrics()
        metrics.acc = current_metrics.acc
        
    def post_epoch(self, epoch, metrics):
        epoch_prefix = f"{epoch:03d}_"
        self.model.eval()        

        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            if key == "acc":
                continue
            metrics.__dict__[key] = value
        metrics.conf_mat = self.conf_mat.to_dict()

        self.evaluate(metrics.__dict__, self.dataset.validation.x, self.dataset.validation.y, "val")
        self.evaluate(metrics.__dict__, self.dataset.test.x, self.dataset.test.y, "test")
        
        acc = metrics.val_acc
        
        if epoch == 0 or epoch == self.config.epochs-1 or acc > np.max(self.history["val_acc"]):
            path = f"{core.output_path}/{epoch_prefix}model_{acc:.4f}.pt"
            print(f"saving model weights to '{path}'")
            torch.save(self.model.state_dict(), path)
        
        if epoch == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 21))
        for i, metric in enumerate(("loss", "acc")):
            for j, prefix in enumerate(("", "val_", "test_")):
                key = f"{prefix}{metric}"
                axes[i, j].set_title(key)
                data = np.asarray(self.history[key])
                if j == 0:
                    data = data.reshape((epoch+1, self.num_mini_batches))
                    if i == 0:
                        data = np.mean(data, axis=1)
                    else:
                        data = data[:,-1]
                else:
                    data = np.concatenate([data, [metrics.__dict__[key]]])
                axes[i, j].plot(data)
                for k in range(1, data.shape[0]):
                    data[k] = (1 - self.config.smoothing)*data[k] + self.config.smoothing*data[k-1]
                axes[i, j].plot(data)
        plt.tight_layout()
        fig.savefig(f"{core.output_path}/history.pdf")
        plt.close(fig)
        
    def finalize(self):
        self.push_notification = ""
        for key, func, header in (("val_loss", np.argmin, "LOSS"), ("val_acc", np.argmax, "ACC")):
            index = func(self.history[key])
            self.push_notification = f"{self.push_notification}--- {header} ---\nepoch = {index+1}\n"
            for metric in ("val_loss", "val_acc", "test_loss", "test_acc"):
                self.push_notification = f"{self.push_notification}{metric} = {self.history[metric][index]:.4f}\n"
            self.push_notification = f"{self.push_notification}\n"
        
    def evaluate(self, metrics, xs, ys, prefix):
        loss = 0
        num_mini_batches = 0
        self.conf_mat.reset()
        
        mbi = utils.get_mini_batch_iterator(self.config.mini_batch_size)
        t = time.perf_counter()
        for x, y in mbi(xs, ys):
            x = torch.from_numpy(x).float().to(core.device)
            yp = self.model(x)
            yt = torch.from_numpy(y).long().to(core.device)
            loss += self.loss_func(yp, yt).item()
            num_mini_batches += 1
            yp = yp.argmax(1).cpu().numpy()
            self.conf_mat.add(y, yp)
        t = time.perf_counter() - t
        metrics[f"{prefix}_time"] = t
        metrics[f"{prefix}_time_per_sample"] = t / xs.shape[0]
        metrics[f"{prefix}_loss"] = loss / num_mini_batches
        
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            metrics[f"{prefix}_{key}"] = value
        metrics[f"{prefix}_conf_mat"] = self.conf_mat.to_dict()
