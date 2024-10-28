/*
 * Copyright 2024 The Ewha Womans University.
 * All Rights Reserved.
 *
 * Permission to use, copy, modify OR distribute this software and its
 * documentation for educational, research, and non-profit purposes, without
 * fee, and without a written agreement is hereby granted, provided that the
 * above copyright notice and the following three paragraphs appear in all
 * copies.
 *
 * IN NO EVENT SHALL THE EWHA WOMANS UNIVERSITY BE
 * LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR
 * CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE
 * USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE EWHA WOMANS UNIVERSITY
 * HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 *
 * THE EWHA WOMANS UNIVERSITY SPECIFICALLY DISCLAIM ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE EWHA WOMANS UNIVERSITY
 * HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
 * UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 *
 * The authors may be contacted via:
 *
 * Mail:        Heajung Min, Kyung Min Han, and Young J. Kim
 *              Computer Graphics Lab                       
 *              Department of Computer Science and Engineering
 *              Ewha Womans University
 *              11-1 Daehyun-Dong Seodaemun-gu, Seoul, Korea 120-750
 *
 * Phone:       +82-2-3277-6798
 *
 * EMail:       hjmin@ewha.ac.kr
 *              hankm@ewha.ac.kr
 *              kimy@ewha.ac.kr
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
