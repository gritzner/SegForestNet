import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import time
import types
import cv2 as cv
import rust
import functools


class SemanticSegmentation(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        assert core.get_object_meta_info(config.model)[0] == "SegForestNet" or not config.visualize_regions
        
        self.dataset = core.create_object(datasets, config.dataset)
        config.num_samples_per_epoch = getattr(config, "num_samples_per_epoch", self.dataset.training.x.shape[0])
        assert 0 < config.num_samples_per_epoch
        assert config.num_samples_per_epoch <= self.dataset.training.x.shape[0]
        
        shape = list(self.dataset.training.x.shape[-3:])
        if self.config.autoencoder:
            shape[0] = 1
        self.model = core.create_object(models, config.model, input_shape=shape, num_classes=self.dataset.num_classes)
        self.model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y: None)
        
        num_params = 0
        for params in self.model.parameters():
            num_params += np.product(params.shape)
        self.history["num_model_parameters"] = int(num_params)
        print(f"# of model parameters: {num_params/10**6:.2f}M")
            
        print(f"# of training samples per epoch: {config.num_samples_per_epoch}\n") # add an extra new line for nicer output formatting
        self.eval_datasets = {}
        for dataset in getattr(config, "evaluation_datasets", []):
            dataset = core.create_object(datasets, dataset)
            assert dataset.training.x.shape[1:] == self.dataset.training.x.shape[1:]
            assert dataset.num_classes == self.dataset.num_classes
            assert dataset.ignore_class == self.dataset.ignore_class
            for i in range(*dataset.split[:2]):
                dataset.base.images[i] = None
            assert not dataset.name in self.eval_datasets
            self.eval_datasets[dataset.name] = dataset
            print()
        
        if not hasattr(config, "epochs"):
            config.epochs = config.model_specific_defaults[self.model.model_name][0]
        if not hasattr(config.learning_rate, "max_value"):
            config.learning_rate.max_value = config.model_specific_defaults[self.model.model_name][1]
        if not hasattr(config, "mini_batch_size"):
            config.mini_batch_size = config.model_specific_defaults[self.model.model_name][2]
        self.num_mini_batches = int(np.ceil(config.num_samples_per_epoch / self.config.mini_batch_size))            
        
        self.loss_func = SemanticSegmentation.binary_ce_loss if config.alt_loss else nn.functional.cross_entropy
        self.optim = utils.optim_wrapper(config.optimizer.type)(
            self.model.parameters(), lr=self.config.learning_rate.max_value,
            **(config.optimizer.arguments.__dict__ if hasattr(config.optimizer,"arguments") else {})
        )
        self.lr_scheduler = utils.LearningRateScheduler(
            self.optim.base_optimizer if hasattr(self.optim, "first_step") else self.optim,
            self.config.learning_rate.min_value,
            self.config.learning_rate.num_cycles, self.config.learning_rate.cycle_length_factor,
            round(self.config.learning_rate.num_iterations_factor * self.num_mini_batches * self.config.epochs)
        )
        
        if config.class_weights.ignore_dataset:
            self.dataset.class_weights[:] = 1
        self.dataset.class_weights[self.dataset.ignore_class] = config.class_weights.ignored_class_weight
            
        self.conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        self.output_set.update(("loss", "acc", "miou", "mf1", "val_loss", "val_acc", "val_miou", "val_mf1"))
        self.shuffle_rng = np.random.RandomState(core.random_seeds[self.config.shuffle_seed])

        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        for image_id in np.unique(self.dataset.validation.x.patch_info[:,0]):
            path = f"{core.output_path}/val_images/{image_id}"
            core.call(f"mkdir -p {path}")
            img = self.dataset.base.images[image_id].base[:,:,self.dataset.base.visualization_channels]
            if img.dtype != np.uint8:
                img = np.asarray(img, dtype=np.uint8)
            cv.imwrite(f"{path}/input.png", np.flip(img, axis=2), (cv.IMWRITE_PNG_COMPRESSION, 9))
            cv.imwrite(f"{path}/gt.png", lut[self.dataset.base.images[image_id].gt], (cv.IMWRITE_PNG_COMPRESSION, 9))
    
    def pre_epoch(self, epoch):
        self.model.train()
        self.model_prepare_func(epoch, self.config.epochs)
        if self.config.unique_iterations:
            if epoch == 0:
                assert self.config.epochs * self.config.num_samples_per_epoch <= self.dataset.training.x.shape[0]
                self.full_shuffle_map = self.shuffle_rng.permutation(self.dataset.training.x.shape[0])
            self.shuffle_map = self.full_shuffle_map[epoch*self.config.num_samples_per_epoch:(epoch+1)*self.config.num_samples_per_epoch]
        else:
            self.shuffle_map = self.shuffle_rng.permutation(self.dataset.training.x.shape[0])[:self.config.num_samples_per_epoch]
        self.conf_mat.reset()
        
    def pre_evaluate(self, epoch):
        self.model.eval()
        self.model_prepare_func(epoch, self.config.epochs)
        self.num_mini_batches = int(np.ceil(self.eval_params.dataset.x.shape[0] / self.config.mini_batch_size))
        self.shuffle_map = np.arange(self.eval_params.dataset.x.shape[0], dtype=np.int32)
        self.eval_params.loss = 0
        self.eval_params.predictions = {
            image_id: [
                np.zeros((*self.eval_params.images[image_id].base.shape[:2], self.dataset.num_classes), dtype=np.float64),
                np.zeros((*self.eval_params.images[image_id].base.shape[:2], 1), dtype=np.uint64)
            ] for image_id in np.unique(self.eval_params.dataset.x.patch_info[:,0])
        }
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        indices = self.shuffle_map[batch*self.config.mini_batch_size:(batch+1)*self.config.mini_batch_size]
        dataset = self.eval_params.dataset if self.eval_params.enabled else self.dataset.training
        batch_data.x = dataset.x_gt[indices] if self.config.autoencoder else dataset.x[indices]
        batch_data.yt = dataset.y[indices]
        if self.eval_params.enabled:
            batch_data.index_map = dataset.index_map[indices]
            if self.eval_params.visualize_regions:
                batch_data.x_vis = dataset.x_vis[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            self.optim.zero_grad()
        x = torch.from_numpy(batch_data.x).float().to(core.device).requires_grad_()
        yt = torch.from_numpy(batch_data.yt).long().to(core.device).detach()
        if not self.eval_params.enabled:
            yp, loss = self.model(x, yt, self.loss_func, self.dataset.class_weights, self.dataset.ignore_class)
            if self.config.gradient_clipping > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
            loss.backward()
            if hasattr(self.optim, "first_step"):
                self.optim.first_step(zero_grad=True)
                yp, loss = self.model(x, yt, self.loss_func, self.dataset.class_weights, self.dataset.ignore_class)
                loss.backward()
                self.optim.second_step(zero_grad=True)
            else:
                self.optim.step()
            metrics.learning_rate = self.lr_scheduler.get_lr()
            self.lr_scheduler.step(epoch * self.num_mini_batches + batch)
            metrics.loss = loss.item()
            batch_data.yp = yp.argmax(1).cpu().numpy()
        else:
            if self.eval_params.visualize_regions:
                path = f"{core.output_path}/region_visualization/{epoch:03d}"
                if batch == 0:
                    core.call(f"mkdir -p {path}")
                path = f"{path}/{batch:08d}"
                yp, loss = self.model(
                    x, yt, self.loss_func, self.dataset.class_weights, self.dataset.ignore_class,
                    batch_data.x_vis, self.eval_params.lut, path
                )
            else:
                yp, loss = self.model(x, yt, self.loss_func, self.dataset.class_weights, self.dataset.ignore_class)
            self.eval_params.loss += loss.item()
            batch_data.yp = yp.softmax(1).cpu().numpy()
        
    def post_train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            self.conf_mat.add(batch_data.yt, batch_data.yp)
            current_metrics = self.conf_mat.compute_metrics()
            metrics.acc = current_metrics.acc
            metrics.miou = current_metrics.miou
            metrics.mf1 = current_metrics.mf1
        else:
            for yp, index_map in zip(batch_data.yp, batch_data.index_map):
                p = self.eval_params.predictions[index_map[0,0,0]]
                p[0][index_map[1],index_map[2]] += np.moveaxis(yp, 0, 2)
                p[1][index_map[1],index_map[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_params.time = time.perf_counter() - self.eval_params.time
        metrics = self.eval_params.metrics.__dict__
        metrics[f"{self.eval_params.prefix}time"] = self.eval_params.time
        metrics[f"{self.eval_params.prefix}time_per_sample"] = self.eval_params.time / self.eval_params.dataset.x.shape[0]
        metrics[f"{self.eval_params.prefix}loss"] = self.eval_params.loss / self.num_mini_batches
        
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        self.conf_mat.reset()
        for image_id, p in self.eval_params.predictions.items():
            yt = self.eval_params.images[image_id].gt
            yt = yt if isinstance(yt,np.ndarray) else yt.get_semantic_image()
            yt = yt if yt.dtype==np.int32 else np.asarray(yt, dtype=np.int32)
            yp = p[0] / p[1]
            ypc = np.argmax(yp, axis=2)
            self.conf_mat.add(
                np.expand_dims(yt, axis=0),
                np.expand_dims(ypc, axis=0)
            )
            rust.prepare_per_pixel_entropy(yp, 10**-6)
            p[0][:,:,0] = np.sum(yp, axis=2)
            if hasattr(self.eval_params, "image_prefix"):
                cv.imwrite(
                    f"{core.output_path}/val_images/{image_id}/{self.eval_params.image_prefix}prediction.png",
                    lut[ypc], (cv.IMWRITE_PNG_COMPRESSION, 9)
                )    
        
        entropy = [0, 0]
        for p, _ in self.eval_params.predictions.values():
            p = p[:,:,0]
            entropy[0] += np.sum(p)
            entropy[1] += np.prod(p.shape)
        metrics[f"{self.eval_params.prefix}entropy"] = float(entropy[0] / entropy[1])
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            metrics[f"{self.eval_params.prefix}{key}"] = value
        metrics[f"{self.eval_params.prefix}conf_mat"] = self.conf_mat.to_dict()
        
    def post_epoch(self, epoch, metrics):
        epoch_prefix = f"{epoch:03d}_"
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            if key == "acc" or key == "miou" or key == "mf1":
                continue
            metrics.__dict__[key] = value
        metrics.conf_mat = self.conf_mat.to_dict()

        self.evaluate(
            epoch, metrics=metrics, dataset=self.dataset.validation, images=self.dataset.base.images,
            prefix="val_", image_prefix=epoch_prefix, visualize_regions=False
        )
        for name, dataset in self.eval_datasets.items():
            self.evaluate(
                epoch, metrics=metrics, dataset=dataset.validation, images=dataset.base.images,
                prefix=f"{name}_val_", visualize_regions=False
            )
        
        path = f"{core.output_path}/{epoch_prefix}model_{metrics.val_miou:.4f}_{metrics.val_mf1:.4f}{self.config.model_filename_extension}"
        print(f"saving model weights to '{path}'")
        with core.open(path, "wb") as f:
            torch.save(self.model.state_dict(), f)
        
        if self.config.class_weights.dynamic_exponent > 0:
            for c in range(self.dataset.num_classes):
                if c == self.dataset.ignore_class:
                    continue
                iou = metrics.__dict__[f"iou{c}"]
                if iou < 0:
                    iou = 0
                self.dataset.class_weights[c] = np.power(1 - iou + metrics.miou, self.config.class_weights.dynamic_exponent)
        
        if epoch == 0:
            return
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 28))
        for i, metric in enumerate(("loss", "acc", "miou", "mf1")):
            for j, prefix in enumerate(("", "val_")):
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
        self.history["test"] = {}
        for i in range(self.dataset.split[0], self.dataset.split[2]):
            self.dataset.base.images[i] = None
        for name, dataset in self.eval_datasets.items():
            self.history[f"{name}_test"] = {}
            for i in range(*dataset.split[1:3]):
                dataset.base.images[i] = None
        
        relevant_epochs = set()
        relevant_epochs.add(len(self.history["val_loss"])-1)
        relevant_epochs.add(np.argmin(self.history["val_loss"]))
        relevant_epochs.add(np.argmin(self.history["val_entropy"]))
        relevant_epochs.add(np.argmax(self.history["val_acc"]))
        relevant_epochs.add(np.argmax(self.history["val_miou"]))
        relevant_epochs.add(np.argmax(self.history["val_mf1"]))
        
        import glob
        for model_fn in sorted(glob.iglob(f"{core.output_path}/*_model_*{self.config.model_filename_extension}")):
            epoch = int(model_fn.split("/")[-1].split("_")[0])
            if not epoch in relevant_epochs:
                if self.config.delete_irrelevant_models:
                    core.call(f"rm {model_fn}")
                continue
            print(f"evaluating '{model_fn}' on test set(s)...")
            with core.open(model_fn, "rb") as f:
                self.model.load_state_dict(torch.load(f, map_location=core.device))
            metrics = types.SimpleNamespace()
            self.evaluate(
                epoch, metrics=metrics, dataset=self.dataset.test, images=self.dataset.base.images, prefix="",
                visualize_regions=self.config.visualize_regions, lut=np.asarray(self.dataset.lut, dtype=np.uint8)
            )
            self.history["test"][epoch] = metrics.__dict__
            for name, dataset in self.eval_datasets.items():
                metrics = types.SimpleNamespace()
                self.evaluate(
                    epoch, metrics=metrics, dataset=dataset.test, images=dataset.base.images,
                    prefix="", visualize_regions=False
                )
                self.history[f"{name}_test"][epoch] = metrics.__dict__

    @staticmethod
    def binary_ce_loss(yp, yt, weight, ignore_index):
        loss = []
        target = torch.empty(yt.shape, dtype=torch.float32, device=yp.device)
        for c in range(yp.shape[1]):
            target[:] = 0
            if c != ignore_index:
                target[yt==c] = 1
            loss.append(weight[c] * torch.nn.functional.binary_cross_entropy_with_logits(yp[:,c], target.clone().detach()))
        return functools.reduce(lambda x,y: x+y, loss)
                