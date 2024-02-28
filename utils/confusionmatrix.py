import numpy as np
import types
import rust


class ConfusionMatrix():
    def __init__(self, num_classes, ignore_class):
        assert num_classes > 1
        self.C = np.empty([num_classes, num_classes], dtype=np.uint64)
        self.ignore_class = ignore_class
    
    def reset(self):
        self.C[:] = 0
    
    def add(self, yt, yp):
        rust.add_to_conf_mat(self.C, yt, yp)
    
    def compute_metrics(self):
        result = {}
        
        for c in range(self.C.shape[0]):
            tp = self.C[c,c]
            fp = np.sum(self.C[:,c]) - tp
            if self.ignore_class >= 0 and c != self.ignore_class:
                fp -= self.C[self.ignore_class,c]
            fn = np.sum(self.C[c,:]) - tp
            
            result[f"iou{c}"] = float(tp / max(tp+fp+fn,1))
            result[f"p{c}"] = float(tp / max(tp+fp,1))
            result[f"r{c}"] = float(tp / max(tp+fn,1))
            result[f"f1_{c}"] = float((2*tp) / max(2*tp+fp+fn,1))
        
        result["acc"] = float(np.sum(np.diag(self.C)) / max(np.sum(self.C),1))
        result["miou"] = float(np.mean([result[f"iou{c}"] for c in range(self.C.shape[0]) if c != self.ignore_class]))
        result["mp"] = float(np.mean([result[f"p{c}"] for c in range(self.C.shape[0]) if c != self.ignore_class]))
        result["mr"] = float(np.mean([result[f"r{c}"] for c in range(self.C.shape[0]) if c != self.ignore_class]))
        result["mf1"] = float(np.mean([result[f"f1_{c}"] for c in range(self.C.shape[0]) if c != self.ignore_class]))
        
        return types.SimpleNamespace(**result)
    
    def to_dict(self):
        return {c: [int(self.C[c,c2]) for c2 in range(self.C.shape[1])] for c in range(self.C.shape[0])}
    
    @staticmethod
    def from_dict(C, ignore_class):
        conf_mat = ConfusionMatrix(len(C), ignore_class)
        for c, cs in C.items():
            c = int(c)
            assert 0 <= c < len(C)
            assert type(cs) == list
            assert len(cs) == len(C)
            for c2, n in enumerate(cs):
                assert type(n) == int
                assert 0 <= n
                conf_mat.C[c,c2] = n
        return conf_mat
