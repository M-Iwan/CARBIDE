# Clone and enter the repository
git clone https://github.com/jrwnter/cddd.git
cd cddd

# Remove defaults from the environment file 
sed -i 's/defaults/conda-forge/g' environment.yml

# Build base environment with CPU support only
conda env create -f environment.yml
conda activate cddd
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.0-cp36-cp36m-linux_x86_64.whl

# Remove install requirements from setup and install CDDD package
sed -i '/install_requires=/,/}/d; /extras_require=/,/}/d' setup.py
pip install .

echo "Open https://github.com/jrwnter/cddd, download the default_model.zip from Google Drive and unpack it in the created cddd directory. Copy the CDDD_wrapper.py to cddd/files, the CDDD_conf.sh to cddd and run it"