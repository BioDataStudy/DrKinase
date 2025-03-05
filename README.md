# Dr. Kinase
## Dr. Kinase: Predicting the drug-resistance mutation hotspots of protein kinases.

Protein kinases (PKs) regulate various cellular functions, and targeted by small-molecule kinase inhibitors (KIs) in cancers and other diseases. However, drug-resistance (DR) of KIs through critical mutations in four types of representative hotspots, including gatekeeper, G-loop, αC-helix, and A-loop. KI drug-resistance has become a common clinical complication affecting multiple cancers, targeted kinases and drugs. To tackle on this challenge, we report an upgraded webserver, namely Dr. Kinase, for predicting the loci of four DR hotspots for PKs upon our previous studies, by utilizing multimodal features and deep hybrid learning. The performance of Dr. Kinase has been rigorously evaluated using independent testing, demonstrating excellent accuracy with area under the curve (AUC) values exceeding 0.89 in different types of DR hotspot predictions. We further conducted in silico analyses to evaluate and validate the epidermal growth factor receptor mutations on protein conformation and KIs-binding efficacy. Dr. Kinase is freely available at http://modinfor.com/drkinase, with comprehensive annotations and visualizations. We anticipate that Dr. Kinase will be a highly useful service for basic, translational, and clinical community to unveil the molecular mechanisms of DR and the develop of next-generation KIs for emerging cancer precision medicine.

<div align=center><img src="http://modinfor.com/MetaDegron/images/index_workflow.png" width="800px"></div>

# Installation
Download Dr. Kinase by
```
git clone https://github.com/BioDataStudy/DrKinase.git
```
Installation has been tested in Linux server, CentOS Linux release 7.8.2003 (Core), with Python 3.7. Since the package is written in python 3x, python3x with the pip tool must be installed. Dr. Kinase uses the following dependencies: numpy, scipy, pandas, h5py, torch, allennlp, keras version=2.3.1, tensorflow=1.15 shutil, and pathlib. We highly recommend that users leave a message under the MetaDegron issue interface (https://github.com/BioDataStudy/DrKinase/issue) when encountering any installation and running problems. We will deal with it in time. You can install these packages by the following commands:
```
conda create -n DrKinase python=3.7
conda activate DrKinase
pip install pandas
pip install numpy
pip install scipy
pip install torch
pip install allennlp==0.9.0
pip install -v keras==2.3.1
pip install -v tensorflow==1.15
pip install seaborn
pip install shutil
pip install protobuf==3.20
pip install h5py==2.10.0
```

# Usage

### Please download large protein language model from the http://modinfor.com/MetaDegron/Download/weights.hdf5. Note: right click - > save the link
### Copy or move the weights.hdf5 file to the models\uniref50_v2\ directory to ensure that the model is preloaded successfully

Please cd to the folder which contains DrKinase_Prediction.py.
```
python DrKinase_prediction.py -inf 'test/test.txt' -mof 'Gatekeeper' 'G-loop' 'A-loop' 'αC-Helix' -out 'prediction/'
```
For details of other parameters, run:
```
python DrKinase_Prediction.py --help
```

