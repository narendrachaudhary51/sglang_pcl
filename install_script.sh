
source /swtools/intel/2025.2.0/setvars.sh --force > /dev/null 2>&1
export UV_CONFIG_FILE=$HOME/sglang_pcl/.venv/uv.toml

source .venv/bin/activate

# Use dedicated toml file
cd python
rm -rf ~/.triton/cache/*
find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
rm -rf build/ sglang_cpu.egg-info/
cp pyproject_cpu.toml pyproject.toml
uv pip install --upgrade pip setuptools
uv pip install .

# Build the CPU backend kernels
cd ../sgl-kernel
cp pyproject_cpu.toml pyproject.toml
uv pip install .
