FROM nvcr.io/nvidia/pytorch:22.09-py3

RUN git clone https://github.com/NVIDIA/FasterTransformer.git && \
  mkdir -p FasterTransformer/build && \
  cd FasterTransformer/build && \
  git submodule init && \
  git submodule update

RUN cd FasterTransformer/build && \
  # DSM=80 is specific to the A100, see https://github.com/NVIDIA/FasterTransformer/blob/main/docs/t5_guide.md
  cmake -DSM=80 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=OFF .. && \
  make -j

# Install mambaforge and mamba

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#   bash /tmp/miniconda.sh -b -u -p /opt/conda && \
#   rm /tmp/miniconda.sh && \
#   ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#   echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
#   echo "conda activate base" >> ~/.bashrc && \
#   /opt/conda/bin/conda install -y mamba

# TODO combine into previous command next time I re-run
# RUN echo "conda activate ar-inference" >> ~/.bashrc


# Create a conda environment from the environment.yml file
# COPY environment.yml /app/inference/environment.yml
# RUN mamba env create -f /app/inference/environment.yml

# Just have it sleep forever
CMD ["sleep", "infinity"]
