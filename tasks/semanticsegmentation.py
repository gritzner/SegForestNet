import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import time
import tasks


class SemanticSegmentation(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        
        prev_task = params.previous_tasks[-1][0] if hasattr(params, "previous_tasks") else None
        prev_type = type(prev_task)
        
        if prev_type == tasks.SemanticSegmentation:
            self.config.autoencoder = prev_task.config.autoencoder
            self.config.mini_batch_size = prev_task.config.mini_batch_size
            self.dataset = prev_task.dataset
            self.num_mini_batches = prev_task.num_mini_batches
            self.model = prev_task.model
            self.loss_func = prev_task.loss_func
        else:
            self.dataset = core.create_object(datasets, config.dataset)
        
            shape = list(self.dataset.training.x.shape[-3:])
            if self.config.autoencoder:
                shape[0] = 1
                
            region_encoder = None
            self.model = core.create_object(models.segmentation, config.model, input_shape=shape, num_classes=self.dataset.num_classes, region_encoder=region_encoder)

            num_params = 0
            for params in self.model.parameters():
                num_params += np.product(params.shape)
            self.history["num_model_parameters"] = int(num_params)
            print(f"# of model parameters: {num_params/10**6:.2f}M")
            
            print() # add a new line for nicer output formatting
        
            if not hasattr(config, "mini_batch_size"):
                config.mini_batch_size = config.model_specific_defaults[self.model.model_name][2]
            self.num_mini_batches = int(np.ceil(self.dataset.training.x.shape[0] / self.config.mini_batch_size))
            
            if self.config.ignore_model_loss or not hasattr(self.model, "get_loss_function"):
                self.loss_func = nn.CrossEntropyLoss(weight=self.dataset.class_weights, ignore_index=self.dataset.ignore_class)
            else:
                self.loss_func = self.model.get_loss_function(self.dataset.class_weights, self.dataset.ignore_class)
        
        if not hasattr(config, "epochs"):
            config.epochs = config.model_specific_defaults[self.model.model_name][0]
        if not hasattr(config.learning_rate, "max_value"):
            config.learning_rate.max_value = config.model_specific_defaults[self.model.model_name][1]
        
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate.max_value)
        self.lr_scheduler = utils.LearningRateScheduler(
            self.optim, self.config.learning_rate.min_value,
            self.config.learning_rate.num_cycles, self.config.learning_rate.cycle_length_factor,
            self.num_mini_batches * self.config.epochs
        )
        
        self.conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        self.output_set.update(("loss", "acc", "miou", "val_loss", "val_acc", "val_miou"))
        self.shuffle_rng = np.random.RandomState(core.random_seeds[self.config.shuffle_seed])
        self.fixed_output_val_images, self.output_images = SemanticSegmentation.initialize_validation_image_grid(
            self.config.val_sample_images, self.shuffle_rng, self.dataset.validation.y.shape, 3
        )
        
    def pre_epoch(self, epoch):
        self.model.train()
        self.shuffle_map = self.shuffle_rng.permutation(self.dataset.training.x.shape[0])
        self.conf_mat.reset()
        
    def pre_evaluate(self, epoch):
        self.num_mini_batches = int(np.ceil(self.eval_params.dataset.x.shape[0] / self.config.mini_batch_size))
        self.shuffle_map = np.arange(self.eval_params.dataset.x.shape[0], dtype=np.int32)
        self.conf_mat.reset()
        self.eval_params.loss = 0
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        indices = self.shuffle_map[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        dataset = self.eval_params.dataset if self.eval_params.enabled else self.dataset.training
        batch_data.x = dataset.x_gt[indices] if self.config.autoencoder else dataset.x[indices]
        batch_data.yt = dataset.y[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            self.optim.zero_grad()
        x = torch.from_numpy(batch_data.x).float().to(core.device).requires_grad_()
        yp = self.model(x)
        yt = torch.from_numpy(batch_data.yt).long().to(core.device)
        loss = self.loss_func(yp, yt)
        if self.eval_params.enabled:
            self.eval_params.loss += loss.item()
        else:
            loss.backward()
            self.optim.step()
            metrics.learning_rate = self.lr_scheduler.get_lr()
            self.lr_scheduler.step(epoch * self.num_mini_batches + batch)
            metrics.loss = loss.item()
        batch_data.yp = yp.argmax(1).cpu().numpy()
        
    def post_train(self, epoch, batch, batch_data, metrics):
        self.conf_mat.add(batch_data.yt, batch_data.yp)
        if not self.eval_params.enabled:
            current_metrics = self.conf_mat.compute_metrics()
            metrics.acc = current_metrics.acc
            metrics.miou = current_metrics.miou
        
    def post_evaluate(self, epoch):
        self.eval_params.time = time.perf_counter() - self.eval_params.time
        metrics = self.eval_params.metrics.__dict__
        metrics[f"{self.eval_params.prefix}_time"] = self.eval_params.time
        metrics[f"{self.eval_params.prefix}_time_per_sample"] = self.eval_params.time / self.eval_params.dataset.x.shape[0]
        metrics[f"{self.eval_params.prefix}_loss"] = self.eval_params.loss / self.num_mini_batches
        
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            metrics[f"{self.eval_params.prefix}_{key}"] = value
        metrics[f"{self.eval_params.prefix}_conf_mat"] = self.conf_mat.to_dict()
        
    def post_epoch(self, epoch, metrics):
        epoch_prefix = f"{epoch:03d}_"
        self.model.eval()
        
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            if key == "acc" or key == "miou":
                continue
            metrics.__dict__[key] = value
        metrics.conf_mat = self.conf_mat.to_dict()

        self.evaluate(epoch, metrics=metrics, dataset=self.dataset.validation, prefix="val")
        self.evaluate(epoch, metrics=metrics, dataset=self.dataset.test, prefix="test")
        
        acc = metrics.val_acc
        miou = metrics.val_miou
        
        if epoch == 0 or epoch == self.config.epochs-1 or epoch == self.config.terminate_early-1 or acc > np.max(self.history["val_acc"]) or miou > np.max(self.history["val_miou"]):
            path = f"{core.output_path}/{epoch_prefix}model_{acc:.4f}_{miou:.4f}.pt"
            print(f"saving model weights to '{path}'")
            torch.save(self.model.state_dict(), path)
        
        indices = np.empty(self.config.val_sample_images.amount.total, dtype=np.int32)
        with torch.no_grad():
            indices[:self.config.val_sample_images.amount.fixed] = self.fixed_output_val_images
            indices[self.config.val_sample_images.amount.fixed:] = np.random.randint(self.dataset.validation.x.shape[0], size=self.config.val_sample_images.amount.random)
            x = self.dataset.validation.x_gt[indices] if self.config.autoencoder else self.dataset.validation.x[indices]
            x = torch.from_numpy(x).float().to(core.device)
            yp = self.model(x).argmax(1).cpu().numpy()
            for i, index in enumerate(indices):
                self.output_images.set_image(0, i, self.dataset.validation.x_vis[index])
                self.output_images.set_image(1, i, self.dataset.validation.y[index], self.dataset.lut)
                self.output_images.set_image(2, i, yp[i], self.dataset.lut)
        path = f"{core.output_path}/{epoch_prefix}images.png"
        print(f"saving sample images to '{path}'")
        self.output_images.save(path)
        
        if hasattr(self.loss_func, "region_loss_images"):
            # assumption: this attribute only exists during debugging of the model's region loss
            import functools
            image_grid = utils.ImageGrid(self.dataset.validation.y.shape[-2:], (2*(len(self.loss_func.region_loss_images)+1), indices.shape[0]))
            with torch.no_grad():
                x = self.dataset.validation.x_gt[indices] if self.config.autoencoder else self.dataset.validation.x[indices]
                y = self.dataset.validation.y[indices]
                x = torch.from_numpy(x).float().to(core.device)
                yp = self.model(x)
                yt = torch.from_numpy(y).long().to(core.device)
                loss = self.loss_func(yp, yt)
                loss_images = [np.asarray(255 * loss_image / np.max(loss_image), dtype=np.uint8) for loss_image in self.loss_func.region_loss_images]
                for i, j in enumerate(indices):
                    image_grid.set_image(0, i, self.dataset.validation.x_vis[j])
                    image_grid.set_image(1, i, self.dataset.validation.y[j], self.dataset.lut)
                    for k, loss_image in enumerate(loss_images):
                        region_map = self.model.trees[k].region_map.clone().detach().cpu().numpy()
                        regions = [utils.hsv2rgb(l/region_map.shape[1],1,1) for l in range(region_map.shape[1])]
                        regions = [np.expand_dims(np.asarray(region), (1,2)) for region in regions]
                        regions = [region*region_map[i,l] for l, region in enumerate(regions)]
                        regions = functools.reduce(lambda x,y: x+y, regions)
                        regions = np.asarray(regions, dtype=np.uint8)
                        image_grid.set_image(2*(k+1)  , i, regions)
                        image_grid.set_image(2*(k+1)+1, i, loss_image[i])
            image_grid.save(f"{core.output_path}/{epoch_prefix}region_loss.png")

        if epoch == 0:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(21, 21))
        for i, metric in enumerate(("loss", "acc", "miou")):
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
        for key, func, header in (("val_loss", np.argmin, "LOSS"), ("val_acc", np.argmax, "ACC"), ("val_miou", np.argmax, "MIOU")):
            index = func(self.history[key])
            self.push_notification = f"{self.push_notification}--- {header} ---\nepoch = {index+1}\n"
            for metric in ("val_loss", "val_acc", "val_miou", "test_loss", "test_acc", "test_miou"):
                self.push_notification = f"{self.push_notification}{metric} = {self.history[metric][index]:.4f}\n"
            self.push_notification = f"{self.push_notification}\n"
            
    @staticmethod
    def initialize_validation_image_grid(config, rng, validation_shape, rows):
        config.amount.total = config.amount.fixed + config.amount.random
        return rng.permutation(validation_shape[0])[:config.amount.fixed], utils.ImageGrid(validation_shape[-2:],(rows,config.amount.total),**config.grid_params.__dict__)
        