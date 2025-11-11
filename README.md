# Install

Some information on the AWS instance type used for training:
- Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 24.04) 64-bit (x86)
- g4dn.2xlarge
- 60GB of gp3 storage

There is a `requirements.txt` file that was generated from installing the
necessary packages into the AWS pytorch environment.


These are the packages that were explicitly installed to create the
`requirements.txt` file

```
pip install opencv-python
pip install scikit-learn
pip install lightning
pip install timm
pip install -U tensorboardX
```
