import core
import numpy as np
import torch
import torch.nn as nn
import utils


class VQIdentity(nn.Module):
    def __init__(self, **_ignored):
        super().__init__()
        self.loss = (0,)
    
    def prepare_for_epoch(self, epoch, epochs):
        pass
    
    def forward(self, x, *_ignored):
        return x
    
    def get_assignment(self, x):
        raise RuntimeError("unsupported vector quantization type for instance ID prediction")


class VQCodebook(nn.Module):
    def __init__(self, normalized_length, codebook_size, feature_size):
        super().__init__()
        assert codebook_size > 0
        self.normalized_length = normalized_length
        if feature_size > 0:
            self.codebook = nn.Parameter(torch.randn((codebook_size, feature_size), dtype=torch.float32, device=core.device))
        
    def forward(self, x):
        if self.normalized_length > 0:
            x = self.normalized_length * nn.functional.normalize(x, dim=1)
            if hasattr(self, "codebook"):
                w = torch.linalg.vector_norm(self.codebook, dim=1)
                w[w<10**-12] = 10**-12
                self.codebook.div(w.unsqueeze(1))
                self.codebook.mul(self.normalized_length)
        return x        


class VQEuclid(nn.Module):
    def __init__(self, normalized_length, codebook_size, feature_size, loss_weights, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(normalized_length, codebook_size, feature_size)
        self.loss_weights = loss_weights
    
    def prepare_for_epoch(self, epoch, epochs):
        pass
    
    def forward(self, x, *_ignored):
        x = self.codebook(x)
        z = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        d = torch.linalg.vector_norm(z.unsqueeze(1) - self.codebook.codebook.unsqueeze(0), dim=2)
        q = self.codebook.codebook[d.argmin(1)]
        q = q.view(x.shape[0], *x.shape[2:], x.shape[1]).permute(0, 3, 1, 2)
        self.loss = torch.linalg.vector_norm(x.detach()-q,dim=1).mean(), torch.linalg.vector_norm(x-q.detach(),dim=1).mean()
        self.loss = self.loss_weights[0]*self.loss[0], self.loss_weights[1]*self.loss[1]
        return q

    def get_assignment(self, x):
        d = torch.linalg.vector_norm(x.unsqueeze(0) - self.codebook.codebook.unsqueeze(2), dim=1)
        d, i = d.sort(dim=0)
        return i[0], -d[0]
        

class VQSoftEuclid(nn.Module):
    def __init__(self, normalized_length, codebook_size, feature_size, temperature, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(normalized_length, codebook_size, feature_size)
        self.softmax_temperature = utils.get_scheduler(temperature)
        self.loss = (0,)
    
    def prepare_for_epoch(self, epoch, epochs):
        self.softmax_temperature(epoch, epochs)
    
    def forward(self, x, *_ignored):
        x = self.codebook(x)
        z = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        d = torch.linalg.vector_norm(z.unsqueeze(1) - self.codebook.codebook.unsqueeze(0), dim=2)
        d = nn.functional.softmax(-d/self.softmax_temperature.value, dim=1)
        q = d @ self.codebook.codebook
        q = q.view(x.shape[0], *x.shape[2:], x.shape[1]).permute(0, 3, 1, 2)
        return q
        
    def get_assignment(self, x):
        d = torch.linalg.vector_norm(x.unsqueeze(0) - self.codebook.codebook.unsqueeze(2), dim=1)
        d, i = d.sort(dim=0)
        return i[0], -d[0]

    
class VQClusterEuclid(nn.Module):
    def __init__(self, normalized_length, threshold, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(normalized_length, 1, 0)
        self.threshold = threshold
        self.loss = (0,)
    
    def prepare_for_epoch(self, epoch, epochs):
        pass
    
    def forward(self, x, *_ignored):
        return self.codebook(x)

    def get_assignment(self, x):
        assignment = torch.empty(x.shape[1], dtype=torch.long, device=core.device)
        confidence = torch.empty(x.shape[1], dtype=torch.float, device=core.device)
        next_id = 0
        i = torch.arange(x.shape[1], device=core.device)
        while i.shape[0] > 0:
            d = torch.linalg.vector_norm(x[:,i[0]].unsqueeze(1) - x[:,i], dim=0)
            j = d < self.threshold
            i, j, d = i[torch.logical_not(j)], i[j], d[j]
            assignment[j] = next_id
            confidence[j] = -d
            next_id += 1
        return assignment, confidence
    

class VQCosine(nn.Module):
    def __init__(self, codebook_size, feature_size, loss_weights, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(1, codebook_size, feature_size)
        self.loss_weights = loss_weights
        
    def prepare_for_epoch(self, epoch, epochs):
        pass
    
    def forward(self, x, *_ignored):
        x = self.codebook(x)
        z = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        d = z @ self.codebook.codebook.t()
        q = self.codebook.codebook[d.argmax(1)]
        self.loss = (1-(z.detach()*q).sum(1)).mean(), (1-(z*q.detach()).sum(1)).mean()
        self.loss = self.loss_weights[0]*self.loss[0], self.loss_weights[1]*self.loss[1]
        q = q.view(x.shape[0], *x.shape[2:], x.shape[1]).permute(0, 3, 1, 2)
        return q
        
    def get_assignment(self, x):
        d = self.codebook.codebook @ x
        d, i = d.sort(dim=0, descending=True)
        return i[0], d[0]


class VQSoftCosine(nn.Module):
    def __init__(self, codebook_size, feature_size, temperature, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(1, codebook_size, feature_size)
        self.softmax_temperature = utils.get_scheduler(temperature)
        self.loss = (0,)
        
    def prepare_for_epoch(self, epoch, epochs):
        self.softmax_temperature(epoch, epochs)
    
    def forward(self, x, *_ignored):
        x = self.codebook(x)
        z = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        d = z @ self.codebook.codebook.t()
        d = nn.functional.softmax(d/self.softmax_temperature.value, dim=1)
        q = d @ self.codebook.codebook
        q = q.view(x.shape[0], *x.shape[2:], x.shape[1]).permute(0, 3, 1, 2)
        return q
        
    def get_assignment(self, x):
        d = self.codebook.codebook @ x
        d, i = d.sort(dim=0, descending=True)
        return i[0], d[0]


class VQClusterCosine(nn.Module):
    def __init__(self, threshold, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(1, 1, 0)
        self.threshold = threshold
        self.loss = (0,)
    
    def prepare_for_epoch(self, epoch, epochs):
        pass
    
    def forward(self, x, *_ignored):
        return self.codebook(x)

    def get_assignment(self, x):
        assignment = torch.empty(x.shape[1], dtype=torch.long, device=core.device)
        confidence = torch.empty(x.shape[1], dtype=torch.float, device=core.device)
        next_id = 0
        i = torch.arange(x.shape[1], device=core.device)
        while i.shape[0] > 0:
            d = (x[:,i[0]] @ x[:,i])
            j = d > self.threshold
            i, j, d = i[torch.logical_not(j)], i[j], d[j]
            assignment[j] = next_id
            confidence[j] = d if d.dtype==torch.float else d.float()
            next_id += 1
        return assignment, confidence


class VQGumbel(nn.Module):
    def __init__(self, normalized_length, codebook_size, feature_size, hard, temperature, loss_weight, **_ignored):
        super().__init__()
        self.codebook = VQCodebook(normalized_length, codebook_size, feature_size)
        self.gumbel_softmax = lambda x,y: nn.functional.gumbel_softmax(x,tau=y,hard=hard)
        self.softmax_temperature = utils.get_scheduler(temperature)
        self.target_probability = 1 / codebook_size
        self.loss_weight = utils.get_scheduler(loss_weight)
        
    def prepare_for_epoch(self, epoch, epochs):
        self.softmax_temperature(epoch, epochs)
        self.loss_weight(epoch, epochs)
    
    def forward(self, x, return_distribution=False):
        x = self.codebook(x)
        
        z = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
        p = self.gumbel_softmax(z, self.softmax_temperature.value)
        if not return_distribution:
            q = p @ self.codebook.codebook
            q = q.view(x.shape[0], *x.shape[2:], -1).permute(0, 3, 1, 2)
        
        p = p.clone()
        p[p<10**-6] = 10**-6
        target = torch.empty_like(p)
        target[:] = self.target_probability
        self.loss = (self.loss_weight.value * nn.functional.kl_div(p.log(), target, reduction="batchmean"),)
        
        return p.view(x.shape[0],*x.shape[2:],-1).permute(0,3,1,2) if return_distribution else q
    
    def get_assignment(self, x):
        x, i = x.sort(dim=0, descending=True)
        return i[0], x[0]-x[1]
    

class DelayedVQ(nn.Module):
    def __init__(self, true_vq_layer, vq_type, epochs, init_samples, kmeans_iterations):
        super().__init__()
        assert epochs > 0
        self.true_vq_layer = true_vq_layer
        self.delay_only = vq_type in (3, 6)
        self.use_cosine = vq_type in (4, 5)
        self.epochs = epochs
        self.init_samples = init_samples
        self.kmeans_iterations = kmeans_iterations
        self.loss = (0,)
        self.store_features = False
        self.apply_vq = False
        
    def prepare_for_epoch(self, epoch, epochs):
        self.true_vq_layer.prepare_for_epoch(epoch, epochs)
        if not self.training:
            self.store_features = False
        elif epoch == self.epochs-1:
            self.training_features = []
            self.store_features = not self.delay_only
        elif epoch == self.epochs:
            if not self.delay_only:
                with torch.no_grad():
                    self.initialize_codebook()
            del self.training_features
            self.store_features = False
            self.apply_vq = True
    
    def forward(self, x, return_distribution=False):
        if self.apply_vq:
            y = self.true_vq_layer.forward(x, return_distribution)
            self.loss = self.true_vq_layer.loss
        else:
            y = self.true_vq_layer.codebook(x)
            if self.store_features:
                self.training_features.append(y.detach().cpu().numpy())
        return y
    
    def get_assignment(self, x):
        return self.true_vq_layer.get_assignment(x)
    
    def initialize_codebook(self):
        z = np.concatenate([np.moveaxis(z, 1, -1).reshape(-1, z.shape[1]) for z in self.training_features], axis=0)
        z = torch.from_numpy(z).to(core.device)
        cb = self.true_vq_layer.codebook.codebook
        
        cb[0] = z[np.random.randint(z.shape[0])]
        d = torch.empty(z.shape[0], dtype=cb.dtype, device=cb.device)
        for i in range(1, cb.shape[0]):
            d2 = DelayedVQ.cosine_similarity(cb[i-1], z) if self.use_cosine else DelayedVQ.euclidean_distance(cb[i-1], z)
            if i == 1:
                d[:] = d2
            else:
                j = d2<d
                d[j] = d2[j]
            cb[i] = z[self.choose_initial_sample(d)]
        
        label = torch.empty(z.shape[0], dtype=torch.int32, device=cb.device)
        for _ in range(self.kmeans_iterations):
            for i in range(cb.shape[0]):
                d2 = DelayedVQ.cosine_similarity(cb[i], z) if self.use_cosine else DelayedVQ.euclidean_distance(cb[i], z)
                if i == 0:
                    d[:] = d2
                    label[:] = i
                else:
                    j = d2<d
                    d[j] = d2[j]
                    label[j] = i
            for i in range(cb.shape[0]):
                j = label==i
                if j.count_nonzero() > 0:
                    cb[i] = z[j].mean(dim=0)
                else:
                    j = self.choose_initial_sample(d)
                    cb[i] = z[j]
                    d[j] = 0
                    label[j] = i
            self.true_vq_layer.codebook(torch.rand(1, z.shape[1])) # normalize codebook
        
    def choose_initial_sample(self, d):
        i = d.argsort(descending=True)[:self.init_samples]
        p = d[i]**2
        p = p / p.sum()
        return np.random.choice(i.cpu().numpy(), p=p.cpu().numpy())

    @staticmethod
    def euclidean_distance(cb_v, z):
        return torch.linalg.vector_norm(cb_v.unsqueeze(0) - z, dim=1)
    
    @staticmethod
    def cosine_similarity(cb_v, z):
        return 1 - (z @ cb_v)

def create_vector_quantization_layer(**params):
    list_of_types = (
        VQIdentity, # 0
        VQEuclid, VQSoftEuclid, VQClusterEuclid, # 1-3
        VQCosine, VQSoftCosine, VQClusterCosine, # 4-6
        VQGumbel # 7; this entry is being referenced twice in the constructor of SegForestNet (models/segmentation/SegForestNet.py)
    )
    
    t = params.pop("type")
    assert t in tuple(range(len(list_of_types)))
    vq_layer = list_of_types[t](**params)
    if "delay" in params and t != 0:
        vq_layer = DelayedVQ(vq_layer, t, **params["delay"].__dict__)
    return vq_layer
