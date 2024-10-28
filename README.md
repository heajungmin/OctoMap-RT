# OctoMap-RT: Fast Probabilistic Volumetric Mapping Using Ray-Tracing GPUs
This repository is for our RA-L 2023 paper 'OctoMap-RT: Fast Probabilistic Volumetric Mapping Using Ray-Tracing GPUs'.

Please visit the [project page](http://graphics.ewha.ac.kr/octomap-rt/) for more information.


## Usage
This project is developed based on Ubuntu 20.04, OctoMap 1.10.0, and Optix SDK 8.0.0.

1. To build the OctoMap library, please follow the instructions in [OctoMap](https://github.com/OctoMap/octomap).

2. To install Optix SDK, please follow the official instructions in [Optix](https://developer.nvidia.com/designworks/optix/download).
```
cd (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
cmake .
make
```

3. Clone the OctoMap-RT repository

4. Insert the OctoMap-RT into the Optix SDK. You can use the Optix SDK build system.
  * Add a directory for OctoMap-RT in (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/CMakeLists.txt
    ```
    add_subdirectory(OctoMap-RT)
    ```   
  
  * Add the OctoMap-RT code in the Optix SDK directory.
    ```
    cd (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/
    mv (your_path)/OctoMap-RT .    
    ```

5. Link OctoMap library
* Check the path for OctoMap library in (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/OctoMap-RT/CMakeLists.txt
  ```
  include_directories((your_path)/octomap/octomap/include)
  link_directories((your_path)/octomap/lib)
  ```

7. Build 
    ```
    cd (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK
    make
    ```

8. Run
    ```
    (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/bin/OctoMap-RT -i geb079_max50m.graph -o map -res 0.1 -max 10.0
    ```
    ```
    (your_path)/NVIDIA-OptiX-SDK-8.0.0-linux64-x86_64/SDK/bin/OctoMap-RT -i fr_campus.graph. -o map -res 0.2 
    ```

9. Visualize the map results such as *.bt or *.ot using [Octovis](https://github.com/OctoMap/octomap).



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
