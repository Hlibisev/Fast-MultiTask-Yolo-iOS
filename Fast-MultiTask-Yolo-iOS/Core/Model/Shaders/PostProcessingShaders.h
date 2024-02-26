//
//  PostProcessingShaders.h
//  postprocessing_yolo
//
//  Created by Anton Fonin on 14.12.2023.
//

#ifndef PostProcessingShaders_h
#define PostProcessingShaders_h

#import <simd/simd.h>

typedef struct FilterParams {
    float confidenceThreshold;
    int stride;
    simd_float2 factor;
    uint numberOfCoords;
    uint numberOfClasses;
    uint numberOfKpts;
    uint taskId;
} FilterParams;


typedef struct MetalBBox {
    uint classId;
    float confidence;
    int x, y, h, w;
    uint numKpts;
    uint taskId; // we have different heads in yolo, this show which head was used for this box prediction
    int kpts[51];  // 51 is max kpts size but really used first numKpts
} BBox;


typedef struct Tensor {
    float values[23];
} Tensor;

#endif /* PostProcessingShaders_h */
