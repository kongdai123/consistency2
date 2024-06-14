conda create -y -n Consistency2 python=3.9
conda activate Consistency2
conda install -y pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d

pip install -r requirements.txt

source ./scripts/setup_nvdiffrast.sh