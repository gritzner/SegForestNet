# SegForestNet
Reference implementation of SegForestNet, a model which predicts binary space partitioning trees to compute a semantic segmentation of aerial images. [The associated paper titled "SegForestNet: Spatial-Partitioning-Based Aerial Image Segmentation" is available on arXiv](https://arxiv.org/abs/2302.01585). Please cite our paper if you use anything from this repository.

```bibtex
@misc{gritzner2024segforestnet,
      title = {SegForestNet: Spatial-Partitioning-Based Aerial Image Segmentation}, 
      author = {Gritzner, Daniel and Ostermann, Jörn},
      publisher = {arXiv},
      year = {2024},
      eprint = {2302.01585},
      archivePrefix = {arXiv},
      primaryClass = {cs.CV}
      doi = {10.48550/ARXIV.2302.01585},
      url = {https://arxiv.org/abs/2302.01585},
      keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences, I.5.4},
}
```
# Results
Our model delivers state-of-the-art performance, even under non optimal training conditions (see paper for details). While other models, e.g., DeepLab v3+, deliver performance on a similar level, SegForestNet is better at predicting small object such as cars properly. It predicts proper rectangles rather than round-ish shapes. Also, car segments which should be disconnected may merge into one larger region when using other models.

Mean $F_1$ scores:

| | Hannover | Buxtehude | Nienburg | Schleswig | Hameln | Vaihingen | Potsdam | Toulouse |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| FCN | 84.9% | 87.7% | 85.5% | 82.6% | 87.8% | 86.6% | 91.3% | 75.8% |
| DeepLab v3+ | __85.7%__ | 88.7% | 86.7% | __83.6%__ | 88.6% | __86.9%__ | __91.5%__ | __77.6%__ |
| __SegForestNet__ | 85.5% | __88.8%__ | 86.2% | 83.0% | __88.7%__ | 86.8% | 91.3% | 74.8% |
| PFNet | 85.4% | 88.4% | 86.3% | 83.2% | 88.4% | 86.8% | __91.5%__ | 75.8% |
| FarSeg | __85.7%__ | 88.5% | __86.8%__ | 82.8% | 88.4% | 86.7% | 91.4% | 75.0% |
| U-Net | 84.3% | 86.7% | 85.5% | 78.5% | 86.8% | 84.2% | 88.6% | 75.2% |
| RA-FCN | 78.5% | 83.1% | 80.0% | 74.6% | 83.9% | 82.6% | 86.6% | 66.9% |

![](samples.png)

# How to run
### Dependencies
The code has been tested on openSUSE Leap 15.4 running the following software:
* cargo 1.74.1 (1.67.1 may also be sufficient)
* cuda 11.6.1
* libtiff 4.5.0
* matplotlib 3.7.0
* numpy 1.23.5
* opencv 4.6.0 
* python 3.10.10
* pytorch 1.13.1
* pyyaml 6.0
* rustc 1.74.1 (1.67.1 may also be sufficient)
* scikit-learn 1.2.1
* scipy 1.10.0
* torchvision 0.14.1

Optional dependencies:
* geotiff 1.7.0
* tifffile 2021.7.2
* timm 0.9.2

### Preparing the training environment (optional)
Using pretrained encoder weights requires executing ```utils/preprocess/model_weights.py``` once to download the necessary model weights (for legacy reasons this will also download weights for another encoder which is no longer used in this codebase). Two of the datasets (DLR Multi-Sensor Land-Cover Classification (MSLCC) and SemCity Toulouse) also require executing the appropriate Python script in ```utils/preprocess/``` once. This is necessary to convert some ```.tif``` files into a format that OpenCV likes. The scripts in ```utils/preprocess/``` need the optional dependencies.

### Running the code
Open a terminal in the directory you cloned this repository into and execute the following command:

```shell
python aethon.py PW potsdam SegForestNet
```

This will use the configuration file ```cfgs/PW.yaml``` to run our framework. Furthermore, you will need a user configuration file called ```~/.aethon/user.yaml```. An example user configuration can be found in ```user.yaml```. The full configuration our framework will parse will be the concatenation of ```core/defaults.yaml``` and ```cfgs/semseg.yaml```. Additionally, all the occurances of ```$N``` in ```cfgs/PW.yaml``` will be replaced by the parameters given in the commandline, e.g., ```$0``` will become ```potsdam``` and ```$1```will become ```SegForestNet```. The example above will run our framework to train our model with on the Potsdam dataset using the first random seed from the array in ```core/random_seeds.npy``` for data augmentation. This is the same random seed we used for the experiments in our paper.

Even though we cannot provide some of the datasets used in the paper for legal reasons we still provide their data loaders as a reference. The data loaders can be found in ```datasets/```.

The training results, including an evaluation of the trained model on the validation and test subsets, can be found in the appropriate subfolder in ```tmp/PW/```once training is complete.

### Running within a Jupyter notebook
You can run our code in Jupyter by simply copying the content of ```aethon.py``` to a notebook and adding the commandline parameters to the second line. Example for the second line:

```python
core.init("PW potsdam SegForestNet")
```

# Model code
If you are only interested in the code of our model, take a look at ```models/SegForest*.py```. The class ```SegForestNet``` implements our model. It uses several helper classes to give our already complicated code some additional structuring. The constructor of our model has two parameters in addition to ```self```:
* ```params``` is an object with the two attributes ```input_shape``` and ```num_classes``` so that the model knows what kind of data to expect. See line 29 ```tasks/semanticsegmentation.py``` for an example.
* ```config``` is an object which is a parsed version of the relevant subset of the configuration file used to run our framework, in particular the section ```SegForestNet_params``` in ```cfgs/PW.yaml``` in the example above. The parsing is done by the ```parse_dict``` function in ```core/__init__.py```.

The ```trees``` subsection of the configuration is of particular interest. It defines the number of trees to predict per block. Each entry of the list ```trees``` will later become an instance of ```models/SegForestTree.py``` with each tree object consisting of a pair of decoders and representing a different tree. The attribute ```graph``` defines the tree structure in terms of components (found in ```models/SegForestComponents.py```). ```eval``` is used to turn ```graph``` into an actual tree object which is technically a security problem. However, the only use cases our framework is supposed to be used in are use cases in which the person triggering the execution of our framework has full system access anyway or at least enough system access to execute arbitrary Python or Rust code. **Note:** this is not the only instance of insecure code in our framework. Examples of valid tree graphs are:
* ```BSPTree(2, Line)```: for a BSP tree of depth two, i.e., a total of three inner nodes and four leaf nodes, using $f_1$ from our paper as signed distance function
* ```BSPTree(2, Circle)```: same as above but using $f_3$ instead of $f_1$
* ```BSPNode(BSPTree(1, Line), Leaf, Line)```: a BSP tree with two inner nodes (the left child of the root node is a BSP tree of depth one while the right child is a leaf node already) and three leaf nodes, using $f_1$ in all inner nodes

The different signed distance functions are defined in the appendix of the paper.

The attribute ```one_tree_per_class``` causes the list of ```trees``` to automatically be expanded such that there is exactly one tree for each class. All trees will use the same configuration, e.g., the same ```graph```. In case multiple trees are defined manually an attribute called  ```outputs``` must defined for each tree. It is a list of integers defining which tree is responsible for predicting the logits of which class. Examples:
* ```[0]``` predict logits for the first class
* ```[1, 2, 4]``` predict logits for classes two, three and five

The union of all ```outputs``` must be the set of all classes and the intersection of ```outputs``` of any two different trees must be empty.

If you want to use SegForestNet outside our framework you need these files:
* models/SegForestNet.py
* models/SegForestTree.py
* models/SegForestTreeDecoder.py
* models/SegForestComponents.py
* models/Xception.py
* models/xception.json.bz2
* utils/\_\_init\_\_.py
* utils/vectorquantization.py

You need to fix several dependencies. To remove the dependency on the core module, replace all instances of ```core.device``` with the appropriate device, usually ```torch.device("cuda:0")```. Add ```import gzip``` to ```Xception.py``` and in line 142 use ```gzip.open(...)``` instead of ```core.open(...)```. Also, in the same line, change the first argument of ```open``` to the path of your downloaded Xception model weights. In ```utils/__init__.py``` comment out lines one to three as well as line five.