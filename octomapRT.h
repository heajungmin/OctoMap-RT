/*
 * OctoMap-RT: Fast Probabilistic Volumetric Mapping Using Ray-Tracing GPUs
 * http://graphics.ewha.ac.kr/octomap-rt/
 *  
 * Copyright(c) 2024, Heajung Min, Kyung Min Han, and Young J. Kim, Ewha Womans University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met :
 *
 *     * Redistributions of source code must retain the above copyright notice, this
 *       list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice,
 *       this list of conditions and the following disclaimer in the documentation
 *       and / or other materials provided with the distribution.
 *     * Neither the name of OctoMap-RT nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <octomap/octomap_timing.h>
#include <octomap/octomap.h>

#include <unordered_map>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <cstring>
#include <cstdlib> 
#include <sstream>

#define AABBS_LEVEL 1 

template<typename T>
struct Vec3 {
    T x;
    T y;
    T z;
};

using Scan = Vec3<float>;
using Scan_pointID = Vec3<uint>;
using Step = Vec3<int>;
using NumInstance = Vec3<uint>;

struct InstanceUnit {
    double unit_minX = 0.0;
    double unit_minY = 0.0;
    double unit_minZ = 0.0;
    double unitSize = 0.0;
};

struct Params {    	
    OptixTraversableHandle handle = 0;
    uint* voxel = nullptr;	
    CUdeviceptr scanBuffer = 0; 
    CUdeviceptr scanBuffer_pointID = 0; 
    uint aabbs_one_instance = 0; 
    uint numXxnumY = 0;
    uint aabbNumPerAxis = 0; 
    uint numInstanceX = 0;
    float3 cameraPosition = {};
};

struct Map {
    octomap::OcTree* tree = nullptr;
    uint totalAabb = 0;
    std::vector<OptixAabb> aabbs;
    std::vector<octomap::OcTreeKey> point_key;	
    std::unordered_map<octomap::ScanNode*, Step> adjustStep;
    octomap::OcTreeKey key_origin = {}; 
    NumInstance numInstance = {};
    InstanceUnit unit = {};
};

struct OctoMapRTState {
	OptixDeviceContext context = 0;
	OptixTraversableHandle gas_handle = 0;    
	OptixProgramGroup raygen_prog_group = 0;
	OptixProgramGroup miss_prog_group = 0;
	OptixProgramGroup hitgroup_prog_group = 0;
	CUdeviceptr d_gas_output_buffer = 0;
	OptixPipeline pipeline = 0;
	OptixModule module = 0;
	OptixShaderBindingTable sbt = {};	
	OptixPipelineCompileOptions pipeline_compile_options = {};
	
	CUdeviceptr d_scanBuffer = 0;
	CUdeviceptr d_scanBuffer_pointID = 0;
	uint* hostData_voxel = 0;
	
	Params params = {};
	Map map = {};
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {};

class RTPointcloud : public octomap::Pointcloud {
public:			
    inline octomap::point3d& getPoint(unsigned int i) {
        if (i < points.size())
            return points[i];
        else {
            OCTOMAP_WARNING("Pointcloud::getPoint index out of range!\n");
            return points.back();
        }
    }

    inline void transform_maxrange(octomath::Pose6D transform, double maxRange) {
        for (unsigned int i = 0; i < points.size(); i++) {
            if (points[i].norm() > maxRange) {
                points[i] = points[i].normalized() * static_cast<float>(maxRange);
            }
            points[i] = transform.transform(points[i]);
        }	  
    } 	
};
