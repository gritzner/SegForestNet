import torch
from torch.nn.functional import relu


class Line():
    num_params = 3
    
    @staticmethod
    def compute(x, sample_points):
        return (x[:,:2] * sample_points).sum(1) - x[:,-1]
    
class Square():
    num_params = 3
    
    @staticmethod
    def compute(x, sample_points):
        t = x[:,:2] - sample_points
        return t.abs().max(1)[0] - x[:,-1]
    
class Circle():
    num_params = 3
    
    @staticmethod
    def compute(x, sample_points):
        t = x[:,:2] - sample_points
        return (t**2).sum(1) - x[:,-1]

class Ellipse():
    num_params = 5
    
    @staticmethod
    def compute(x, sample_points):
        d0 = ((x[:,:2] - sample_points)**2).sum(1).sqrt()
        d1 = ((x[:,2:4] - sample_points)**2).sum(1).sqrt()
        return d0 + d1 - x[:,-1]

class Hyperbola():
    num_params = 5
    
    @staticmethod
    def compute(x, sample_points):
        d0 = ((x[:,:2] - sample_points)**2).sum(1).sqrt()
        d1 = ((x[:,2:4] - sample_points)**2).sum(1).sqrt()
        return (d0 - d1).abs() - x[:,-1]

class Parabola():
    num_params = 5
    
    @staticmethod
    def compute(x, sample_points):
        d0 = ((x[:,:2] - sample_points)**2).sum(1).sqrt()
        d1 = Line.compute(x[:,2:], sample_points)
        return d0 - d1

    
class Leaf():
    def __init__(self, index):
        self.index = index

class LeafNode():
    def __init__(self):
        self.num_params = 0
        self.children = [Leaf]
    
    def apply_add(self, region_map, x, sample_points, region_map_lambda):
        region_map[:,self.indices[0]] += region_map_lambda
    
    def apply_mul(self, region_map, x, sample_points, region_map_lambda):
        region_map[:,self.indices[0]] *= region_map_lambda

    def apply_old(self, region_map, x, sample_points, region_map_lambda):
        region_map[:,self.indices[0]] *= region_map_lambda

    
class BSPNode():
    def __init__(self, left, right, sdf):
        self.sdf = sdf
        self.num_params = sdf.num_params
        self.children = [left, right]
        
    def apply_add(self, region_map, x, sample_points, region_map_lambda):
        t = self.sdf.compute(x, sample_points)
        t = (region_map_lambda * t).unsqueeze(1)
        region_map[:,self.indices[0]] += relu(t)
        region_map[:,self.indices[1]] += relu(-t)
    
    def apply_mul(self, region_map, x, sample_points, region_map_lambda):
        t = self.sdf.compute(x, sample_points)
        t[t.sign()==0] = 10**-6
        t = t.unsqueeze(1)
        region_map[:,self.indices[0]] *= relu(t)
        region_map[:,self.indices[1]] *= relu(-t)
    
    def apply_old(self, region_map, x, sample_points, region_map_lambda):
        t = self.sdf.compute(x, sample_points)
        t[t.sign()==0] = 10**-6
        t = torch.sigmoid(region_map_lambda[1] * t).unsqueeze(1)
        region_map[:,self.indices[0]] *= region_map_lambda[0] * t
        region_map[:,self.indices[1]] *= region_map_lambda[0] * (1-t)
    
def BSPTree(depth, sdf):
    return Leaf if depth==0 else BSPNode(BSPTree(depth-1,sdf), BSPTree(depth-1,sdf), sdf)


class QuadtreeNode():
    def __init__(self, a, b, c, d):
        self.num_params = 2
        self.children = [a, b, c, d]

    def apply_add(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points
        t0 = relu(t[:,0]).unsqueeze(1)
        t1 = relu(-t[:,0]).unsqueeze(1)
        t2 = relu(t[:,1]).unsqueeze(1)
        t3 = relu(-t[:,1]).unsqueeze(1)
        region_map[:,self.indices[0]] += region_map_lambda * t0 * t2
        region_map[:,self.indices[1]] += region_map_lambda * t0 * t3
        region_map[:,self.indices[2]] += region_map_lambda * t1 * t2
        region_map[:,self.indices[3]] += region_map_lambda * t1 * t3
    
    def apply_mul(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points
        t[t.sign()==0] = 10**-3
        t0 = relu(t[:,0]).unsqueeze(1)
        t1 = relu(-t[:,0]).unsqueeze(1)
        t2 = relu(t[:,1]).unsqueeze(1)
        t3 = relu(-t[:,1]).unsqueeze(1)
        region_map[:,self.indices[0]] *= t0 * t2
        region_map[:,self.indices[1]] *= t0 * t3
        region_map[:,self.indices[2]] *= t1 * t2
        region_map[:,self.indices[3]] *= t1 * t3

    def apply_old(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points
        t[t.sign()==0] = 10**-3
        t0 = torch.sigmoid(region_map_lambda[1] * t[:,0]).unsqueeze(1)
        t1 = torch.sigmoid(region_map_lambda[1] * t[:,1]).unsqueeze(1)
        region_map[:,self.indices[0]] *= region_map_lambda[0] * t0 * t1
        region_map[:,self.indices[1]] *= region_map_lambda[0] * t0 * (1-t1)
        region_map[:,self.indices[2]] *= region_map_lambda[0] * (1-t0) * t1
        region_map[:,self.indices[3]] *= region_map_lambda[0] * (1-t0) * (1-t1)

def Quadtree(depth):
    return Leaf if depth==0 else QuadtreeNode(Quadtree(depth-1), Quadtree(depth-1), Quadtree(depth-1), Quadtree(depth-1))


class KDTreeNode():
    def __init__(self, left, right, dim):
        self.dim = dim
        self.num_params = 1
        self.children = [left, right]
    
    def apply_add(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points[:,self.dim].unsqueeze(1)
        t *= region_map_lambda
        region_map[:,self.indices[0]] += relu(t)
        region_map[:,self.indices[1]] += relu(-t)
    
    def apply_mul(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points[:,self.dim].unsqueeze(1)
        t[t.sign()==0] = 10**-6
        region_map[:,self.indices[0]] *= relu(t)
        region_map[:,self.indices[1]] *= relu(-t)
        
    def apply_old(self, region_map, x, sample_points, region_map_lambda):
        t = x - sample_points[:,self.dim].unsqueeze(1)
        t[t.sign()==0] = 10**-6
        t = torch.sigmoid(region_map_lambda[1] * t)
        region_map[:,self.indices[0]] *= region_map_lambda[0] * t
        region_map[:,self.indices[1]] *= region_map_lambda[0] * (1-t)
        
def KDTree(depth, root_dim):
    child_dim = (root_dim + 1) % 2
    return Leaf if depth==0 else KDTreeNode(KDTree(depth-1,child_dim), KDTree(depth-1,child_dim), root_dim)


class DynKDTreeNode():
    def __init__(self, left, right):
        self.num_params = 3
        self.children = [left, right]
        
    def apply_add(self, region_map, x, sample_points, region_map_lambda):
        t = x[:,2] - torch.where(x[:,:2].argmax(1)==0, sample_points[:,0], sample_points[:,1])
        t = (region_map_lambda * t).unsqueeze(1)
        region_map[:,self.indices[0]] += relu(t)
        region_map[:,self.indices[1]] += relu(-t)
    
    def apply_mul(self, region_map, x, sample_points, region_map_lambda):
        t = x[:,2] - torch.where(x[:,:2].argmax(1)==0, sample_points[:,0], sample_points[:,1])
        t[t.sign()==0] = 10**-6
        t = t.unsqueeze(1)
        region_map[:,self.indices[0]] *= relu(t)
        region_map[:,self.indices[1]] *= relu(-t)        

    def apply_old(self, region_map, x, sample_points, region_map_lambda):
        t = x[:,2] - torch.where(x[:,:2].argmax(1)==0, sample_points[:,0], sample_points[:,1])
        t[t.sign()==0] = 10**-6
        t = torch.sigmoid(region_map_lambda[1] * t).unsqueeze(1)
        region_map[:,self.indices[0]] *= region_map_lambda[0] * t
        region_map[:,self.indices[1]] *= region_map_lambda[0] * (1-t)

def DynKDTree(depth):
    return Leaf if depth==0 else DynKDTreeNode(DynKDTree(depth-1), DynKDTree(depth-1))
        