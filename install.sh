
#!/bin/bash

wget -O ./res101.pth https://cloudstor.aarnet.edu.au/plus/s/lTIJF4vrvHCAI31/download

conda create --name adelai
conda activate adelai
pip3 install -r requirements.txt

sudo apt-get install -y libsparsehash-dev
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.0.0

# conda install cudatoolkit=10.2
sudo apt install -y nvidia-cuda-toolkit
