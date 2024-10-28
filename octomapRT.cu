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

 #include <optix.h>

#include "octomapRT.h"
#include <cuda/helpers.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

// A utility function to map a hit position index in [0, Nquery] for
//   pixel (x, y) into the appropriate hit buffer index
static __forceinline__ __device__ uint getHitBufferIndex(uint hitPos, uint3 pixelIdx, uint3 pixelDims) {
  return (hitPos*pixelDims.y + pixelIdx.y)*pixelDims.x + pixelIdx.x;
}

static __forceinline__ __device__ uint pointID(uint key0, uint key1, uint key2) {
	uint instance = params.numXxnumY * uint(floorf(float(key2) / float(params.aabbNumPerAxis))) 
                    + params.numInstanceX * uint(floorf(float(key1) / float(params.aabbNumPerAxis))) 
                    + uint(floorf(float(key0) / float(params.aabbNumPerAxis)));

	instance *= params.aabbs_one_instance;

	uint m_aabbNumPerAxis_2 = params.aabbNumPerAxis * params.aabbNumPerAxis;

	uint primitive = m_aabbNumPerAxis_2 * ((key2) % params.aabbNumPerAxis);
	primitive += params.aabbNumPerAxis * ((key1) % params.aabbNumPerAxis);
	primitive += (key0) % params.aabbNumPerAxis;
	
	return instance + primitive;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint hitBufferIndex = getHitBufferIndex(0, idx, dim);

    Scan r = reinterpret_cast<Scan*>(params.scanBuffer)[hitBufferIndex];
    
    // occupied voxel id
    uint p_id = pointID(
        reinterpret_cast<Scan_pointID*>(params.scanBuffer_pointID)[hitBufferIndex].x, 
        reinterpret_cast<Scan_pointID*>(params.scanBuffer_pointID)[hitBufferIndex].y, 
        reinterpret_cast<Scan_pointID*>(params.scanBuffer_pointID)[hitBufferIndex].z);		
        
    float3 rayDirection = make_float3(r.x, r.y, r.z) - params.cameraPosition;
    
    optixTrace(
        params.handle,
        params.cameraPosition,
        normalize(rayDirection),
        0.00f,  // tmin
        length(rayDirection), //1e16f,  // tmax
        0.0f,                // rayTime
        OptixVisibilityMask( 1 ),
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        0,                   // SBT stride
        0,                   // missSBTIndex
        p_id);
}

extern "C" __global__ void __intersection__is()
{     
    optixReportIntersection( 0, 0, 0, 0 );
}

extern "C" __global__ void __anyhit__ah()
{   
	uint pointID = optixGetInstanceIndex() * params.aabbs_one_instance + optixGetPrimitiveIndex();      
		
    if (pointID == optixGetPayload_0()) 
    { // occupied AABB						
        uint m = 1 << (2 * (pointID % 16));
        uint n = pointID / 16;
        atomicOr(&params.voxel[n], m);			
    }
    else 
    { // free AABB						
        uint m = 1 << (2 * (pointID % 16) + 1);
        uint n = pointID / 16;
        atomicOr(&params.voxel[n], m);
    }        
	
	optixIgnoreIntersection();		
}

extern "C" __global__ void __miss__ms(){}
extern "C" __global__ void __closesthit__ch(){}