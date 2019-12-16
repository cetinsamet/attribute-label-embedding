# attribute-label-embedding  
An Implementation of Attribute Label Embedding (ALE) Method for Zero Shot Learning

### Requirements
python==3.6.9  
pip install -r requirements.txt
  
### Instructions
1. Download missing resnet features of the dataset that you are planning to use, from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/) under proposed splits link. Place downloaded features in corresponding directory under "./datasets".  
2. Run corresponding "./datasets/prepareDATASETNAME.ipynb" script to prepare data for use. 
3. Run corresponding "bash scripts/run_DATASETNAME.sh" command to train ALE method.
