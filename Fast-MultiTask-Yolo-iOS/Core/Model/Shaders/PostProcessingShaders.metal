#include <metal_stdlib>
#include "PostProcessingShaders.h"
using namespace metal;



kernel void filterBoxes(device float *predictions [[buffer(0)]],
                        constant FilterParams &params [[buffer(1)]],
                        device atomic_int* bboxCount [[ buffer(2) ]],
                        device MetalBBox* bboxes [[ buffer(3) ]],
                        uint id [[thread_position_in_grid]],
                        uint idInGroup [[thread_index_in_threadgroup]])
    
{
    int stride = (params.stride + 31) / 32 * 32;  // predictions is texture, and auto added padding to size % 32
    
    float maxConfidence = 0;
    uint classId = 0;
    uint start = params.numberOfCoords;
    uint length = start + params.numberOfClasses;
    
    for (uint i = start; i < length; ++i) {
        float confidence = predictions[id + stride * i];
        if (confidence > maxConfidence) {
            maxConfidence = confidence;
            classId = i - start;
        }
    }
    
    if (maxConfidence > params.confidenceThreshold) {
        int remainder;
        float value;
        
        float x = predictions[id];
        float y = predictions[id + stride];
        float w = predictions[id + stride * 2];
        float h = predictions[id + stride * 3];
        
        MetalBBox box {
            .classId = classId,
            .confidence = maxConfidence,
            .x = int((x - 0.5 * w) * params.factor.x),
            .y = int((y - 0.5 * h) * params.factor.y),
            .w = int(w * params.factor.x),
            .h = int(h * params.factor.y),
            .taskId = params.taskId,
            .numKpts = params.numberOfKpts
        };
        
        uint start = params.numberOfCoords + params.numberOfClasses;
        uint end = start + params.numberOfKpts;
        
        for (uint i = start; i < end; ++i) {
            value = predictions[id + stride * i];
            remainder = i % 3;
            
            if (remainder == 0) {
                box.kpts[i - start] = int(value * params.factor.y);
            } else if (remainder == 1) {
                box.kpts[i - start] = int(value > 0.5);
            } else if (remainder == 2) {
                box.kpts[i - start] = int(value * params.factor.x);
            }
        }
        
        int i = atomic_fetch_add_explicit(bboxCount, 1, memory_order_relaxed);
        bboxes[i] = box;
    }
}
