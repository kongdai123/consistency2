rm -rf ./nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
cd ./nvdiffrast
git checkout c5caf7b
PATCH_PATH=../patches/nvdiffrast.patch
git apply "${PATCH_PATH}"
pip install .
cd ../