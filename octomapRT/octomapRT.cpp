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

#include "octomapRT.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <cuda_runtime.h>
#include <sampleConfig.h>

#include <sutil/sutil.h>
#include <cuda/whitted.h>

#include <sutil/CUDAOutputBuffer.h>

using namespace octomap;
using namespace octomath;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

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
    uint numVoxel = 0;

    Params params = {};
    Map map = {};
};

std::unique_ptr<sutil::CUDAOutputBuffer<uint>> output_voxel;
OctoMapRTState state;

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

OcTree* OctomapRT::getTree() const {
    return state.map.tree;
}

void OctomapRT::sync() {
    CUDA_SYNC_CHECK();
}

OctomapRT::OctomapRT(double res, double maxRange, double clampingMin, double clampingMax, double probHit, double probMiss)
    : res(res), maxRange(maxRange), clampingMin(clampingMin), clampingMax(clampingMax), probHit(probHit), probMiss(probMiss) {    
    state.map.tree = new octomap::OcTree(res);
    state.map.tree->setClampingThresMin(clampingMin);
    state.map.tree->setClampingThresMax(clampingMax);
    state.map.tree->setProbHit(probHit);
    state.map.tree->setProbMiss(probMiss);
}

OctomapRT::~OctomapRT() {    
}

void OctomapRT::initialize(ScanGraph* graph) {
    initAABB(graph);
    initCUDAOptix();
    bottomLevelAS();
    topLevelAS();
    createModule();
    createProgramGroups();
    linkPipeline();
    shaderBindingTable();
}

void OctomapRT::insertPointCloud(ScanGraph::iterator scan_it) {
    initScan(scan_it);
    launch((*scan_it)->scan->size(), 1);
    readback();
    treeUpdate(scan_it);
}

void OctomapRT::initAABB(ScanGraph *graph)
{
    double res_half = res * 0.5;
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    min_x = min_y = min_z = 1e6;
    max_x = max_y = max_z = -1e6;

    octomap::point3d lowerBound(0.0, 0.0, 0.0);
    octomap::point3d upperBound(0.0, 0.0, 0.0);

    // area filled with AABBs
    for (ScanGraph::iterator scan_it = graph->begin(); scan_it != graph->end(); scan_it++)
    {
        pose6d frame_origin = (*scan_it)->pose;
        point3d lower, upper;
        RTPointcloud *pointcloud = static_cast<RTPointcloud *>((*scan_it)->scan);

        double _x = floor(frame_origin.x() / res);
        double _y = floor(frame_origin.y() / res);
        double _z = floor(frame_origin.z() / res);

        float voxelX = static_cast<float>(_x * res); // left bottom
        float voxelY = static_cast<float>(_y * res);
        float voxelZ = static_cast<float>(_z * res);

        // sensor offset in a voxel
        pose6d voxelizedLocal = pose6d(frame_origin.x() - voxelX, frame_origin.y() - voxelY, frame_origin.z() - voxelZ, frame_origin.roll(), frame_origin.pitch(), frame_origin.yaw());

        // adjust point cloud by sensor offset in a voxel
        if (maxRange > 0) {
            pointcloud->transform_maxrange(voxelizedLocal, maxRange);
        } else {
            (*scan_it)->scan->transform(voxelizedLocal);
        }

        point3d sensor_origin = frame_origin.inv().transform((*scan_it)->pose.trans());
        (*scan_it)->pose = pose6d(voxelizedLocal.transform(sensor_origin), octomath::Quaternion());

        Step step_value = {static_cast<int>(_x), static_cast<int>(_y), static_cast<int>(_z)};
        state.map.adjustStep[*scan_it] = step_value;

        pointcloud->calcBBX(lower, upper);

        // adjust bbx including sensor
        if ((*scan_it)->pose.x() < lower.x())
            lower.x() = (*scan_it)->pose.x();
        if ((*scan_it)->pose.y() < lower.y())
            lower.y() = (*scan_it)->pose.y();
        if ((*scan_it)->pose.z() < lower.z())
            lower.z() = (*scan_it)->pose.z();

        if ((*scan_it)->pose.x() > upper.x())
            upper.x() = (*scan_it)->pose.x();
        if ((*scan_it)->pose.y() > upper.y())
            upper.y() = (*scan_it)->pose.y();
        if ((*scan_it)->pose.z() > upper.z())
            upper.z() = (*scan_it)->pose.z();

        float x = lower(0);
        float y = lower(1);
        float z = lower(2);
        float xx = upper(0);
        float yy = upper(1);
        float zz = upper(2);

        if (x < min_x)
            min_x = x;
        if (y < min_y)
            min_y = y;
        if (z < min_z)
            min_z = z;

        if (xx > max_x)
            max_x = xx;
        if (yy > max_y)
            max_y = yy;
        if (zz > max_z)
            max_z = zz;

        lowerBound(0) = min_x;
        lowerBound(1) = min_y;
        lowerBound(2) = min_z;
        upperBound(0) = max_x;
        upperBound(1) = max_y;
        upperBound(2) = max_z;
    }

    // std::cout << "all scan's lowerBound = " << lowerBound(0) << " " << lowerBound(1) << " " << lowerBound(2) << std::endl;
    // std::cout << "all scan's upperBound = " << upperBound(0) << " " << upperBound(1) << " " << upperBound(2) << std::endl;

    auto InitializeAABB = [&](const float3 &pointPosition, const float3 &size) {
        return OptixAabb {
            -size.x + pointPosition.x,
            -size.y + pointPosition.y,
            -size.z + pointPosition.z,
            pointPosition.x + size.x,
            pointPosition.y + size.y,
            pointPosition.z + size.z,
        };
    };

    state.map.aabbs.clear();
    std::vector<OptixAabb>().swap(state.map.aabbs);

    // number of aabbs on one side of an instance
    // length of one side of an instance
    state.map.unit.unitSize = res * static_cast<double>(pow(2, AABBS_LEVEL));

    auto calculateUnit = [&](double bound) {
        return floor(static_cast<double>(bound) / state.map.unit.unitSize) * state.map.unit.unitSize;
    };
    state.map.unit.unit_minX = calculateUnit(lowerBound(0));
    state.map.unit.unit_minY = calculateUnit(lowerBound(1));
    state.map.unit.unit_minZ = calculateUnit(lowerBound(2));
    double unit_maxX = calculateUnit(upperBound(0));
    double unit_maxY = calculateUnit(upperBound(1));
    double unit_maxZ = calculateUnit(upperBound(2));

    auto calculateNumInstances = [&](double max, double min) {
        return static_cast<uint>((max - min) / state.map.unit.unitSize + 1.0);
    };
    state.map.numInstance.x = calculateNumInstances(unit_maxX, state.map.unit.unit_minX);
    state.map.numInstance.y = calculateNumInstances(unit_maxY, state.map.unit.unit_minY);
    state.map.numInstance.z = calculateNumInstances(unit_maxZ, state.map.unit.unit_minZ);

    state.map.totalAabb = state.map.numInstance.x * state.map.numInstance.y * state.map.numInstance.z * pow(pow(2, AABBS_LEVEL), 3);

    // aabb setup
    uint aabbNumPerAxis = pow(2, AABBS_LEVEL);
    for (uint z = 0; z < aabbNumPerAxis; z++) {
        for (uint y = 0; y < aabbNumPerAxis; y++) {
            for (uint x = 0; x < aabbNumPerAxis; x++) {
                float centerX = static_cast<float>(static_cast<double>(x) * res + res_half);
                float centerY = static_cast<float>(static_cast<double>(y) * res + res_half);
                float centerZ = static_cast<float>(static_cast<double>(z) * res + res_half);
                state.map.aabbs.push_back(InitializeAABB(
                    make_float3(centerX, centerY, centerZ),
                    make_float3(static_cast<float>(res_half), static_cast<float>(res_half), static_cast<float>(res_half))));
            }
        }
    }

    state.map.tree->coordToKeyChecked(point3d(state.map.unit.unit_minX + res_half, state.map.unit.unit_minY + res_half, state.map.unit.unit_minZ + res_half), state.map.key_origin); 

    state.params.aabbs_one_instance = (pow(pow(2, AABBS_LEVEL), 3));    
    state.params.numXxnumY = state.map.numInstance.x * state.map.numInstance.y;
    state.params.aabbNumPerAxis = aabbNumPerAxis;
    state.params.numInstanceX = state.map.numInstance.x;
    
    state.numVoxel = (state.map.totalAabb) / 16 + 1;
    output_voxel = std::unique_ptr<sutil::CUDAOutputBuffer<uint>>(
    new sutil::CUDAOutputBuffer<uint>(sutil::CUDAOutputBufferType::CUDA_DEVICE, state.numVoxel, 1));

}

void OctomapRT::initCUDAOptix() {
    state.context = nullptr;

    CUDA_CHECK(cudaFree(0));

    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
}

void OctomapRT::bottomLevelAS() {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    CUdeviceptr d_aabb_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb_buffer), state.map.aabbs.size() * sizeof(OptixAabb)));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_aabb_buffer),
        state.map.aabbs.data(),
        state.map.aabbs.size() * sizeof(OptixAabb),
        cudaMemcpyHostToDevice));

    const size_t num_indices = state.map.aabbs.size();
    std::vector<uint32_t> sbt_index(num_indices);

    for (size_t i = 0; i < num_indices; ++i) {
        sbt_index[i] = static_cast<uint32_t>(i);
    }

    CUdeviceptr d_sbt_index;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_sbt_index), num_indices * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_sbt_index), sbt_index.data(), num_indices * sizeof(uint32_t), cudaMemcpyHostToDevice));

    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
    aabb_input.customPrimitiveArray.numPrimitives = state.map.aabbs.size();

    std::vector<uint32_t> aabb_input_flags(state.map.aabbs.size(), OPTIX_GEOMETRY_FLAG_NONE);
    aabb_input.customPrimitiveArray.flags = aabb_input_flags.data();
    aabb_input.customPrimitiveArray.numSbtRecords = state.map.aabbs.size();
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_sbt_index;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0, // CUDA stream
                                &accel_options,
                                &aabb_input,
                                1, // num build inputs
                                d_temp_buffer_gas,
                                gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size,
                                gas_buffer_sizes.outputSizeInBytes,
                                &state.gas_handle,
                                &emitProperty, // emitted property list
                                1              // num emitted properties
                                ));

    CUDA_CHECK(cudaFree((void *)d_temp_buffer_gas));
    CUDA_CHECK(cudaFree((void *)d_aabb_buffer));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_sbt_index)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void *)d_buffer_temp_output_gas_and_compacted_size));
    } else {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void OctomapRT::topLevelAS() {    
    OptixTraversableHandle ias_handle = 0; // Traversable handle for instance AS
    CUdeviceptr d_ias_output_buffer = 0;   // Instance AS memory

    const size_t num_instances = state.map.numInstance.x * state.map.numInstance.y * state.map.numInstance.z;
    std::vector<OptixInstance> optix_instances(num_instances);

    size_t instance_index = 0;
    double res_half = res * 0.5;
    float eps = static_cast<float>(res_half);

    for (uint z = 0; z < state.map.numInstance.z; z++) {
        for (uint y = 0; y < state.map.numInstance.y; y++) {
            for (uint x = 0; x < state.map.numInstance.x; x++) {
                float xx = static_cast<float>(state.map.unit.unit_minX + static_cast<double>(x) * state.map.unit.unitSize);
                float yy = static_cast<float>(state.map.unit.unit_minY + static_cast<double>(y) * state.map.unit.unitSize);
                float zz = static_cast<float>(state.map.unit.unit_minZ + static_cast<double>(z) * state.map.unit.unitSize);

                optix_instances[instance_index].traversableHandle = state.gas_handle;
                optix_instances[instance_index].flags = OPTIX_INSTANCE_FLAG_NONE;
                optix_instances[instance_index].instanceId = static_cast<unsigned int>(instance_index);
                optix_instances[instance_index].sbtOffset = 0;
                optix_instances[instance_index].visibilityMask = 1;

                float transform[12] = {
                    1.0f, 0.0f, 0.0f, xx,
                    0.0f, 1.0f, 0.0f, yy,
                    0.0f, 0.0f, 1.0f, zz};
                std::memcpy(optix_instances[instance_index].transform, transform, sizeof(transform));

                instance_index++;

                for (auto &it : state.map.aabbs) {
                    float pointX = it.minX + xx;
                    float pointY = it.minY + yy;
                    float pointZ = it.minZ + zz;

                    OcTreeKey key_point;
                    state.map.tree->coordToKeyChecked(point3d(pointX + eps, pointY + eps, pointZ + eps), key_point);

                    state.map.point_key.push_back(key_point); // store key for all AABBs
                }
            }
        }
    }

    CUdeviceptr d_instances;
    size_t instance_size_in_bytes = sizeof(OptixInstance) * num_instances;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_instances), instance_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_instances), optix_instances.data(), instance_size_in_bytes, cudaMemcpyHostToDevice));

    OptixBuildInput instance_input = {};
    instance_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>(num_instances);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &instance_input, 1, &ias_buffer_sizes));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer), ias_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_ias_output_buffer), ias_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accel_options,
        &instance_input,
        1, // num build inputs
        d_temp_buffer,
        ias_buffer_sizes.tempSizeInBytes,
        d_ias_output_buffer,
        ias_buffer_sizes.outputSizeInBytes,
        &ias_handle,
        nullptr, // emitted property list
        0        // num emitted properties
        ));

    state.params.handle = ias_handle;

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_instances)));
}

void OctomapRT::createModule() {
    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 1;
    state.pipeline_compile_options.numAttributeValues = 0;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t inputSize = 0;    
    const char *input = sutil::getInputData("octomapRT", "octomapRT", "octomapRT.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreate(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.module));
}

void OctomapRT::createProgramGroups() {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        LOG, &LOG_SIZE,
        &state.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        LOG, &LOG_SIZE,
        &state.miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        LOG, &LOG_SIZE,
        &state.hitgroup_prog_group));
}

void OctomapRT::linkPipeline() {
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, state.pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));
}

void OctomapRT::shaderBindingTable() {
    CUdeviceptr raygen_record;
    size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;
}

void OctomapRT::initScan(ScanGraph::iterator& scan_it) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_scanBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_scanBuffer_pointID)));

    size_t numPointcloud = (*scan_it)->scan->size();

    std::vector<Scan> scanBuffer(numPointcloud);
    std::vector<Scan_pointID> scanBuffer_pointID(numPointcloud);

    RTPointcloud *pc = static_cast<RTPointcloud *>((*scan_it)->scan);

    for (int i = 0; i < (int)pc->size(); i++) {
        point3d &p = pc->getPoint(i);
        OcTreeKey key_point = {0, 0, 0};

        if (state.map.tree->coordToKeyChecked(p.x(), p.y(), p.z(), key_point)) {
            key_point[0] -= state.map.key_origin[0];
            key_point[1] -= state.map.key_origin[1];
            key_point[2] -= state.map.key_origin[2];

            scanBuffer[i] = Scan{p.x(), p.y(), p.z()};
            scanBuffer_pointID[i] = Scan_pointID{key_point.k[0], key_point.k[1], key_point.k[2]}; 
        }
    }

    state.params.cameraPosition = {(*scan_it)->pose.x(), (*scan_it)->pose.y(), (*scan_it)->pose.z()};

    size_t scanBuffer_size = numPointcloud * sizeof(Scan);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_scanBuffer), scanBuffer_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_scanBuffer_pointID), scanBuffer_size));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_scanBuffer), scanBuffer.data(), scanBuffer_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_scanBuffer_pointID), scanBuffer_pointID.data(), scanBuffer_size, cudaMemcpyHostToDevice));
}

void OctomapRT::launch(uint width, uint height) {
    CUstream stream = 0;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaMemset(output_voxel->map(), 0, state.numVoxel * sizeof(uint)));

    state.params.voxel = output_voxel->map();
    state.params.scanBuffer = state.d_scanBuffer;                 // ray endpoint
    state.params.scanBuffer_pointID = state.d_scanBuffer_pointID; // occupied voxel id

    CUdeviceptr d_param = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &state.params, sizeof(state.params),
        cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(state.pipeline, stream, d_param, sizeof(Params), &state.sbt, width, height, 1));

    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    output_voxel->unmap();
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));
}

void OctomapRT::readback() {
    state.hostData_voxel = output_voxel->getHostPointer();
}

void OctomapRT::treeUpdate(ScanGraph::iterator& scan_it) {
    // read 2 bits
    for (uint c = 0; c < state.numVoxel; c++) {
        bool lazy_eval = false;
        uint i = state.hostData_voxel[c]; // 32bit

        for (uint v = 0; v < 16; v++) { // per 2bit
            uint option1 = 1 << (2 * v);
            uint option2 = 1 << (2 * v + 1);
            uint new_c = c * 16 + v;

            if (i & option1) { // occupied
                OcTreeKey key = state.map.point_key[new_c];
                Step step_value = state.map.adjustStep[*scan_it];
                key[0] += step_value.x;
                key[1] += step_value.y;
                key[2] += step_value.z;
                state.map.tree->updateNode(key, true, lazy_eval);
            }
            else if (i & option2) { // free
                OcTreeKey key = state.map.point_key[new_c];
                Step step_value = state.map.adjustStep[*scan_it];
                key[0] += step_value.x;
                key[1] += step_value.y;
                key[2] += step_value.z;
                state.map.tree->updateNode(key, false, lazy_eval);
            }
        }
    }
}

void OctomapRT::cleanup() {
    output_voxel.reset();

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_gas_output_buffer)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    delete state.map.tree;
}