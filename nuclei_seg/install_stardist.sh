micromamba env create -n stardist python=3.10 -y

micromamba activate stardist

pip install reikna==0.8

pip install "tensorflow[and-cuda]<2.18"

pip install gputools "numpy<2.0"

pip install stardist pandas