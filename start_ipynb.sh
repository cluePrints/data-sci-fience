export PATH="$PATH:/home/ubuntu/anaconda2/bin:/usr/local/cuda/bin:/snap/bin:/home/ubuntu/bin:/home/ubuntu/.local/bin"
jupyter notebook --notebook-dir=/home/ubuntu/downloads --profile=nbserver > /tmp/ipynb.out 2>&1 &
