ROOM_TYPE=$1
TAG=$2
DEVICE=$3

export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

CUDA_VISIBLE_DEVICES=$DEVICE \

for exp in {0..0}; do
  lr="5e-4"
  seed=$((0 + exp))
  echo ">>> Running with lr = $lr, seed = $seed"
  CUDA_VISIBLE_DEVICES=$DEVICE \
    python3 src/test_sg2sc_image.py configs/${ROOM_TYPE}.yaml \
      --tag $TAG --lr $lr --seed $seed
done