import core
import datasets
import models
import numpy as np
import torch
import torch.nn as nn
import utils
import time
import cv2 as cv


#TODO: may be broken and in need of fixing
class EvalSemanticSegmentationModel(utils.ModelFitter):
    def __init__(self, config, params):
        super().__init__(config)
        
        self.dataset = core.create_object(datasets, config.dataset)
        print() # add a new line for nicer output formatting
        
        self.num_mini_batches = 1
        self.config.epochs = 1
        if config.ignore_dataset_class_weights:
            self.dataset.class_weights[:] = 1
            self.dataset.class_weights[self.dataset.ignore_class] = 0
        
        self.conf_mat = utils.ConfusionMatrix(self.dataset.num_classes, self.dataset.ignore_class)
        self.conf_mat.reset()
        self.output_set.update(("val_loss", "val_acc", "val_miou", "test_loss", "test_acc", "test_miou"))

        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        core.call(f"mkdir -p {core.output_path}/val_images")
        for image_id in np.unique(self.dataset.validation.x.patch_info[:,0]):
            img = self.dataset.base.images[image_id].base[:,:,self.dataset.validation.x_vis.channels]
            if img.dtype != np.uint8:
                img = np.asarray(img, dtype=np.uint8)
            cv.imwrite(f"{core.output_path}/val_images/{image_id}_input.png", np.flip(img, axis=2), (cv.IMWRITE_PNG_COMPRESSION, 9))
            cv.imwrite(f"{core.output_path}/val_images/{image_id}_gt.png", lut[self.dataset.base.images[image_id].gt], (cv.IMWRITE_PNG_COMPRESSION, 9))
    
    def pre_evaluate(self, epoch):
        self.model_prepare_func(*self.eval_params.config.model_epochs, True)
        self.num_mini_batches = int(np.ceil(self.eval_params.dataset.x.shape[0] / self.eval_params.config.mini_batch_size))
        self.eval_params.loss = 0
        self.eval_params.predictions = {
            image_id: [
                np.zeros((*self.dataset.base.images[image_id].base.shape[:2], self.dataset.num_classes), dtype=np.float64),
                np.zeros((*self.dataset.base.images[image_id].base.shape[:2], 1), dtype=np.uint64)
            ] for image_id in np.unique(self.eval_params.dataset.x.patch_info[:,0])
        }
        self.eval_params.time = time.perf_counter()
        
    def pre_train(self, epoch, batch, batch_data):
        if not self.eval_params.enabled:
            return
        indices = slice(batch * self.eval_params.config.mini_batch_size, (batch+1) * self.eval_params.config.mini_batch_size)
        batch_data.x = self.eval_params.dataset.x_gt[indices] if self.eval_params.config.autoencoder else self.eval_params.dataset.x[indices]
        batch_data.yt = self.eval_params.dataset.y[indices]
        batch_data.index_map = self.eval_params.dataset.index_map[indices]
    
    def train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            return
        with torch.no_grad():
            x = torch.from_numpy(batch_data.x).float().to(core.device)
            yt = torch.from_numpy(batch_data.yt).long().to(core.device).detach()
            yp = self.model(x)
            loss = self.loss_func(yp, yt, weight=self.dataset.class_weights, ignore_index=self.dataset.ignore_class)
            self.eval_params.loss += loss.item()
            batch_data.yp = yp.softmax(1).cpu().numpy()
        
    def post_train(self, epoch, batch, batch_data, metrics):
        if not self.eval_params.enabled:
            return
        for yp, index_map in zip(batch_data.yp, batch_data.index_map):
            p = self.eval_params.predictions[index_map[0,0,0]]
            p[0][index_map[1],index_map[2]] += np.moveaxis(yp, 0, 2)
            p[1][index_map[1],index_map[2]] += 1
        
    def post_evaluate(self, epoch):
        self.eval_params.time = time.perf_counter() - self.eval_params.time
        self.eval_params.metrics[f"{self.eval_params.prefix}time"] = self.eval_params.time
        self.eval_params.metrics[f"{self.eval_params.prefix}time_per_sample"] = self.eval_params.time / self.eval_params.dataset.x.shape[0]
        self.eval_params.metrics[f"{self.eval_params.prefix}loss"] = self.eval_params.loss / self.num_mini_batches
       
        lut = np.flip(np.asarray(self.dataset.lut, dtype=np.uint8), axis=1)
        self.conf_mat.reset()
        for image_id, p in self.eval_params.predictions.items():
            yt = self.dataset.base.images[image_id].gt
            yt = yt if isinstance(yt,np.ndarray) else yt.get_semantic_image()
            yt = yt if yt.dtype==np.int32 else np.asarray(yt, dtype=np.int32)
            yp = np.argmax(p[0] / p[1], axis=2)
            self.conf_mat.add(
                np.expand_dims(yt, axis=0),
                np.expand_dims(yp, axis=0)
            )
            if self.eval_params.save_predicitions:
                cv.imwrite(
                    f"{core.output_path}/val_images/{self.eval_params.config.key}/{image_id}_prediction.png",
                    lut[yp], (cv.IMWRITE_PNG_COMPRESSION, 9)
                )
        
        for key, value in self.conf_mat.compute_metrics().__dict__.items():
            self.eval_params.metrics[f"{self.eval_params.prefix}{key}"] = value
        self.eval_params.metrics[f"{self.eval_params.prefix}conf_mat"] = self.conf_mat.to_dict()
        
    def finalize(self):
        for config in self.config.models:
            print(f"evaluating configuration '{config.key}' ...")
            
            self.dataset.replace_normalization_params(np.load(f"{core.output_path}/normalization_params.npy"))
            if hasattr(config, "normalization_parameters"):
                norm_params = np.load(config.normalization_parameters.path)
                if config.normalization_parameters.std_only:
                    norm_params = norm_params[:,1]
                self.dataset.replace_normalization_params(norm_params)
            
            shape = list(self.dataset.training.x.shape[-3:])
            if config.autoencoder:
                shape[0] = 1
            self.model = core.create_object(models.segmentation, config.model, input_shape=shape, num_classes=self.dataset.num_classes)
            with core.open(config.model_weights, "rb") as f:
                self.model.load_state_dict(torch.load(f, map_location=core.device))
            self.model.eval()
            
            if config.ignore_model_loss or not hasattr(self.model, "get_loss_function"):
                self.loss_func = nn.functional.cross_entropy
            else:
                self.loss_func = self.model.get_loss_function()
            self.model_prepare_func = getattr(self.model, "prepare_for_epoch", lambda x,y,z: None)
            
            metrics = {}
            core.call(f"mkdir -p {core.output_path}/val_images/{config.key}")
            self.evaluate(
                0, metrics=metrics, config=config,
                dataset=self.dataset.validation, prefix="val_", save_predicitions=True
            )
            self.evaluate(
                0, metrics=metrics, config=config,
                dataset=self.dataset.test, prefix="test_", save_predicitions=False
            )
            self.history[config.key] = metrics
