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