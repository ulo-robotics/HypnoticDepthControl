# GPU image: docker build -t lethe .
#            docker run --rm --gpus all -v $PWD/runs:/app/runs lethe python train.py --out runs/lethe
FROM nvcr.io/nvidia/jax:26.05-py3
RUN pip install --no-cache-dir optax scipy matplotlib
WORKDIR /app
COPY . .
