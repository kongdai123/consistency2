rm -rf ./nvdiffrast
git clone https://github.com/NVlabs/nvdiffrast
cd ./nvdiffrast
PATCH_PATH=../patches/nvdiffrast.patch
git apply "${PATCH_PATH}"
pip install .
cd ../