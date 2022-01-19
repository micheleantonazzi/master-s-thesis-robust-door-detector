* # Robust door detection in autonomous mobile robots

  This repository contains my master's thesis, entitled *Robust door detection in autonomous mobile robots*, downloadable [here](https://drive.google.com/file/d/1xI0iMlpUpPJBaG8Ywakc6d2vFn9dc9Ux/view?usp=sharing). This project aims to build a door detector for autonomous agents and to improve its accuracy by exploiting the conditions in which an indoor robot operates. The main idea is to use the *wayfinding* principle, according to which an indoor environment should present a coherent design that helps people to safely and efficiently navigate in an indoor environment. Doors represent a critical feature of an indoor space, and usually, they are similar to each other in the same environment. In its deployment, an autonomous agent operates in a single environment, which is unknown during the development phase. Following this intuition, a door detector used by a mobile robot can be specialized for a certain environment, improving its performance. 

  [download dataset](https://drive.google.com/file/d/1BqjBpobjKTomFjDkzhWjmCryAXOEluO2/view?usp=sharing)

  [download models parameters](https://mega.nz/folder/8kZXDC4D#XyT1IEwQQLAizo_gXNaiyQ)

  ## Data collection

  ### Simulator

  The data collection phase is a fundamental part of this thesis. The doors' dataset is collected using [Gibson](http://gibsonenv.stanford.edu/). Gibson is a virtualization framework that simulates 3D environments and some different robot types (e.g. Turtlebot robot). Since Gibson does not have the necessary features to collect a well-formed image dataset, I contributed to this project implementing the necessary functionalities. My upgrades include adding a new simulation environment without physical constraints, improving the resource management (environments dataset and robot models), improving the build procedure, and setting up a continuous integration workflow using Github Actions. You can find the new version of Gibson [here](https://github.com/micheleantonazzi/GibsonEnv). The simulated worlds come from [Stanford2D3Ds](https://github.com/alexsax/2D-3D-Semantics) and [Matterport3D](https://niessner.github.io/Matterport/), two datasets of real environments semantically annotated.

  ### Pose selector

  The simulated robots in Gibson can't use the standard navigation stack, because the Matterport3D worlds present irregular floors. To solve this issue, a method is proposed in this thesis to extract plausible positions that a robot can reach to collect data. First of all, a 2D map is extracted from each world mesh. This map is a 2D occupancy grid, where white pixels represent free space and black pixels indicate obstacles. The image map is then processed using computer vision techniques and finally, the Voronoi Diagram is computed to delineate a graph. This graph indicates a possible path followed by the robot during exploration. The method's detail and implementation can be found in the [gibson-env-utilities](https://github.com/micheleantonazzi/gibson-env-utilities) package.

  ### Dataset framework

  The dataset of this thesis is composed of RGB images. I developed a configurable framework, called [generic dataset](https://github.com/micheleantonazzi/generic-dataset), to manage a dataset of any king. Using this utility, the programmer can create his own dataset manager according to his needs. In addition, it also offers useful utility to manipulate *numpy arrays*. This utility builds a pipeline (a series of operations to modify an array) which can also be easily run on GPU without modifying the code. For this reason, this library is particularly suitable for image datasets or for those datasets that massively use numpy arrays.

  ## Doors detector

  [DETR](https://arxiv.org/abs/2005.12872) is used to build the doors detector. It is a Transformers-based deep learning module. More specifically, DETR uses a fully end-to-end object detector, which at first extracts feature vectors from images using a CNN backbone. Then, the features are used as input for a classic transformer to find their relationships. Finally, its outputs are post-processed by a linear regressor and a small multi-layer perceptron, whose infer category labels and the bounding boxes, respectively. 

  ## Incremental learning

  As mentioned before, this thesis presents a new technique for increasing the doors detector's performance, called **incremental learning**. It works as follow:

  - At first, a set of 10 different indoor environments *E = {e<sub>0</sub>, ..., e<sub>9</sub>}* is considered.
  - Then, the doors dataset is collected from these 10 environment. It is composed of RGB images. It is composed of RGB images, that may or may not contain a door. The images are divided by the environment they belong to. 
  - For each environment *e<sub>i</sub>*, an instance of DETR is trained using only images from other environments and it is tested using the 25% of the images belonging to the *i-th* world. This first experiment simulates the situation where a general door detector, is trained using a heterogeneous dataset and used by a mobile robot in an unknown environment during the deployment phase.
  - Finally, each general classifier, trained without data from environment *e<sub>i</sub>*, is incrementally fine-tuned using the images belonging to the world *ei* and tested using the same 25% of samples. Each general detector is fine-tuned using the 25%, 50% and 75% of samples from the environment not included in the initial training phase. In this way, the general classifiers are adapted to the specific environment, in which an autonomous agent executes its task. Through this technique is also possible to estimates how the performance how performance increases depending on the percentage of samples used for fine-tuning the model.
