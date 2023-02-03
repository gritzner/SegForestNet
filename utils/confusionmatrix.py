import numpy as np
import types
import rust


class ConfusionMatrix():
    def __init__(self, num_classes, ignore_class):
        self.C = np.empty([num_classes, num_classes], dtype=np.uint64)
        self.ignore_class = ignore_class
    
    def reset(self):
        self.C[:] = 0
    
    def add(self, yt, yp):
        rust.add_to_conf_mat(self.C, yt, yp)
    
    def compute_metrics(self):
        result = {}
        acc = [0, 0]
        ious = []
        
        for c in range(self.C.shape[0]):
            tp = self.C[c,c]
            row_sum = np.sum(self.C[c,:])
            col_sum = np.sum(self.C[:,c])
            
            if c != self.ignore_class:
                acc[0] += tp
                acc[1] += row_sum
                if self.ignore_class >= 0:
                    col_sum -= self.C[self.ignore_class,c]
            
            denom = row_sum + col_sum - tp
            if denom > 0:
                iou = tp / denom
                if c != self.ignore_class:
                    ious.append(iou)
                result[f"iou{c}"] = float(iou)
            else:
                result[f"iou{c}"] = -1
            
            p = tp / col_sum if col_sum>0 else -1
            r = tp / row_sum if row_sum>0 else -1
            f1 = (2 * p * r) / (p + r) if tp>0 else -1
            
            result[f"p{c}"] = float(p)
            result[f"r{c}"] = float(r)
            result[f"f1_{c}"] = float(f1)
            
        result["acc"] = float(acc[0] / max(acc[1],1))
        result["miou"] = float(np.mean(ious)) if len(ious)>0 else -1
        
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
