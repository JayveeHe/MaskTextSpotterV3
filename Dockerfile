FROM centos:7
#WORKDIR /root/MaskTextSpotterV3
#COPY . /root/MaskTextSpotterV3/
RUN yum -y install python3 python3-devel git mesa-libGL centos-release-scl which && \
    yum -y install devtoolset-8-gcc devtoolset-8-gcc-c++ && source /opt/rh/devtoolset-8/enable bash && yum clean all && \
    pip3 install --upgrade pip && \
    git clone https://github.com/JayveeHe/MaskTextSpotterV3.git && cd MaskTextSpotterV3 && \
    source /opt/rh/devtoolset-8/enable bash && \
    pip3 install -r requirements.txt && \
    git clone https://github.com/NVIDIA/apex.git && cd apex && export TORCH_CUDA_ARCH_LIST="compute capability" && python3 setup.py install && cd .. && rm -rf apex &&\
    export MAX_JOBS=1 && python3 setup.py build install && \
    rm -rf /root/.cache/pip

