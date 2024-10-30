# OctoMap-RT: Fast Probabilistic Volumetric Mapping Using Ray-Tracing GPUs
This repository is for our RA-L 2023 paper 'OctoMap-RT: Fast Probabilistic Volumetric Mapping Using Ray-Tracing GPUs'.

Please visit the [project page](http://graphics.ewha.ac.kr/octomap-rt/) for more information.


## Usage
This project is developed based on Ubuntu 20.04, OctoMap 1.10.0, and Optix SDK 8.0.0.

1. To build the OctoMap library, please follow the instructions in [OctoMap](https://github.com/OctoMap/octomap).

2. To install Optix SDK, please follow the official instructions in [Optix](https://developer.nvidia.com/designworks/optix/download).
```
cd ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
cmake .
make
```

3. Clone the OctoMap-RT repository

4. Insert the OctoMap-RT into the Optix SDK. You can use the Optix SDK build system and utils.
  * Add two directories for OctoMap-RT in ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/CMakeLists.txt
    ```
    add_subdirectory(octomapRT)
    add_subdirectory(octomapRT_example)
    ```   
  
  * Add the OctoMap-RT code in the Optix SDK directory.
    ```
    cd ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/
    mv ~/Downloads/OctoMap-RT/octomapRT .
    mv ~/Downloads/OctoMap-RT/octomapRT_example .    
    ```

5. Link OctoMap library
* Check the path for OctoMap library in ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/octomapRT/CMakeLists.txt
* Check the path for OctoMap library in ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/octomapRT_example/CMakeLists.txt
  
  ```
  include_directories(~/Downloads/octomap/octomap/include)
  link_directories(~/Downloads/octomap/lib)
  ```

6. Build 
    ```
    cd ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
    cmake .
    make
    ```

7. Run
* [OctoMap dataset](http://ais.informatik.uni-freiburg.de/projects/datasets/octomap/)
  
    ```
    ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/bin/octomapRT_example -i geb079_max50m.graph -o map -res 0.1 -max 10.0
    ```
    ```
    ~/Downloads/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/bin/octomapRT_example -i fr_campus.graph. -o map -res 0.2 
    ```

8. Visualize the map results such as *.bt or *.ot using [Octovis](https://github.com/OctoMap/octomap).



## Citation
If you use OctoMap-RT in a scientific publication, please cite our paper.
* [IEEE](https://ieeexplore.ieee.org/document/10197524)

```
@article{min2023octomap,
  title={OctoMap-RT: Fast probabilistic volumetric mapping using ray-tracing GPUs},
  author={Min, Heajung and Han, Kyung Min and Kim, Young J},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```


## Questions
Please contact 'hjmin@ewha.ac.kr'

## License
This project is licensed under the BSD-2-Clause License. See the [LICENSE](LICENSE) file for details.
