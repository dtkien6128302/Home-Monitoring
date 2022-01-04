# core

Core module for Home Security Solution

Kalman Filter tracking (people)

# fall-detection

## Anaconda

1. Download and install
    * https://docs.anaconda.com/anaconda/install/windows/

2. Create environment
    * conda create --name project-name python=3.6

3. Install necessary packages (should be in this correct order)
    * conda install python=3.6
    * conda install tensorflow-gpu==1.14.0
    * pip install opencv-python==3.4.5.20
    * pip install scipy pyyaml ipykernel matplotlib

## Posenet (windows 10)

1. Download and install Posenet
    * git clone https://www.github.com/rwightman/posenet-python
    * Create videos folder in posenet-python to save sample videos
    * Create images folder in posenet-python to save sample images
    * Create output folder in posenet-python to save results

2. Run code
    * python image_demo.py --model 101 --image_dir ./images --output_dir ./images
    * python webcam_demo.py --model 101 --file ./videos/video.mp4

3. Merge posenet-python to repository architecture
    * Copy the code in webcam_demo.py file and paste into main.py file
    * Replace app folder with posenet folder
    * Replace helper folder with converter folder (might be done previously by replacing app folder)
    * Copy videos folder, images folder and output folder then paste to fall-detection folder
    * Create .env-sample file inside posenet folder

## Posenet (jetson nano - ubuntu 18.04)

1. Install necessary packages
    * mkdir ${HOME}/internship
    * cd ${HOME}/internship
    * git clone https://github.com/jkjung-avt/jetson_nano.git
    * cd jetson_nano
    * ./install_basics.sh
    * source ${HOME}/.bashrc
    * sudo apt update
    * sudo apt install -y build-essential make cmake cmake-curses-gui \
                        git g++ pkg-config curl libfreetype6-dev \
                        libcanberra-gtk-module libcanberra-gtk3-module \
                        python3-dev python3-pip
    * sudo pip3 install -U pip==20.2.1 Cython testresources setuptools
    * cd ${HOME}/internship/jetson_nano
    * ./install_protobuf-3.8.0.sh
    * sudo pip3 install numpy==1.16.1 matplotlib==3.2.2
    * sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
                        zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    * sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 \
                        keras_preprocessing==1.1.1 keras_applications==1.0.8 \
                        gast==0.2.2 futures pybind11
    * sudo pip3 install --pre --extra-index-url \
                        https://developer.download.nvidia.com/compute/redist/jp/v44 \
                        tensorflow==1.15.2

2. Download and install Posenet
    * cd ${HOME}/internship
    * git clone https://gitlab.com/beamlab/hss/fall-detection.git
    * cd fall-detection
    * git pull origin kien

3. Run code
    * python3 main.py --model 101 --file ./videos/video.mp4

## Openpose (jetson nano - ubuntu 18.04)

1. Install necessary packages
    * mkdir ${HOME}/internship
    * cd ${HOME}/internship
    * git clone https://github.com/jkjung-avt/jetson_nano.git
    * cd jetson_nano
    * ./install_basics.sh
    * source ${HOME}/.bashrc
    * sudo apt update
    * sudo apt install -y build-essential make cmake cmake-curses-gui \
                        git g++ pkg-config curl libfreetype6-dev \
                        libcanberra-gtk-module libcanberra-gtk3-module \
                        python3-dev python3-pip
    * sudo pip3 install -U pip==20.2.1 Cython testresources setuptools
    * cd ${HOME}/internship/jetson_nano
    * ./install_protobuf-3.8.0.sh
    * sudo pip3 install numpy==1.16.1 matplotlib==3.2.2
    * sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
                        zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    * sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 \
                        keras_preprocessing==1.1.1 keras_applications==1.0.8 \
                        gast==0.2.2 futures pybind11
    * sudo pip3 install --pre --extra-index-url \
                        https://developer.download.nvidia.com/compute/redist/jp/v44 \
                        tensorflow==1.15.2

    === Install swig & requirements.txt ===
    * cd ${HOME}/internship/jetson_nano
    * sudo apt-get install g++
    * sudo apt-get install libpcre3 libpcre3-dev
    * cd ${HOME}
    ** Download swig from: http://www.swig.org/download.html
    * cd ${HOME}/Downloads
    * chmod 777 swig-4.0.2.tar.gz
    * tar -xzvf swig-4.0.2.tar.gz
    ** Cut & paste swig-4.0.2 folder (after extracted) into jetson_nano folder
    * cd ${HOME}/internship/jetson_nano/swig-4.0.2
    * ./configure --prefix=/home/seniorproject/internship/jetson_nano/swigtool
    * sudo make
    * sudo make install
    * sudo vim /etc/profile
    * export SWIG_PATH=/home/seniorproject/internship/jetson_nano/swigtool/bin
    * export PATH=$SWIG_PATH:$PATH
    * source /etc/profile
    * swig -version
    ===              Done               ===

2. Download and install Openpose
    * cd ${HOME}/internship
    * git clone https://gitlab.com/beamlab/hss/fall-detection.git
    * cd fall-detection
    * git pull origin pose-pc

3. Run code
    * python3 main.py --camera videos/dance.mp4 --tensorrt True

### Necessary command

* git pull origin kien
* git stash
* git fetch --all
* git checkout kien

* pip3 install -r requirements.txt

#### Reference

1. https://www.geeksforgeeks.org/posenet-pose-estimation/
2. https://github.com/rwightman/posenet-python
3. https://gitlab.com/beamlab/hss/fall-detection.git
