//
//  PostProcesser.swift
//  postprocessing_yolo
//
//  Created by Anton Fonin on 14.12.2023.
//

import Foundation
import MetalKit
import Metal
import CoreML



struct BBox {
    let classId: UInt32
    let confidence: Float
    let x: Int32
    let y: Int32
    let h: Int32
    let w: Int32
    let numKpts: UInt32
    let taskId: UInt32
    var kpts: [Int32] = Array(repeating: 0, count: 51)
}


protocol PostProcessor {
    func apply(prediction: MLMultiArray, factor: simd_float2, numberOfClasses: Int, confidenceThreshold: Float, numKpts: Int, taskId: Int) throws -> [BBox]?
}



class PostProcesserGPU: PostProcessor{
    private var device: MTLDevice
    private var commandQueue: MTLCommandQueue
    private var pipelineKernel: MTLComputePipelineState
    private var library: MTLLibrary
    private var bufferManager: BufferManager
    
    var bboxes: MTLBuffer?
    
    enum Error: Swift.Error {
        case errorAllocateBuffers
    }
    
    init(){
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        library = device.makeDefaultLibrary()!
        pipelineKernel = MTLPipelineState.createCompute(
            library: library, device: device, functionName: "filterBoxes"
        )
        bufferManager = BufferManager(device: device)
        
        allocate()
    }
    func allocate(){
        bboxes = device.makeBuffer(length: MemoryLayout<Tensor>.stride * Int(ObjectDetectionModel.stride))
    }
    
    func allocate_temp() -> MTLBuffer? {
        let bboxCount = device.makeBuffer(length: MemoryLayout<Int32>.stride)
        return bboxCount
    }
    
    func apply(prediction: MLMultiArray, factor: simd_float2 = .one, numberOfClasses: Int = 1, confidenceThreshold: Float = 0.3, numKpts: Int = 0, taskId: Int = 0) throws -> [BBox]? {
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let commandEncoder = commandBuffer.makeComputeCommandEncoder(),
              let predictionBuffer = bufferManager.buffer(for: prediction),
              let bboxes,
              let bboxCount = allocate_temp()
        else { throw Error.errorAllocateBuffers }
                
        var params = FilterParams(
            confidenceThreshold: confidenceThreshold,
            stride: ObjectDetectionModel.stride,
            factor: factor,
            numberOfCoords: UInt32(4),
            numberOfClasses: UInt32(numberOfClasses),
            numberOfKpts: UInt32(numKpts),
            taskId: UInt32(taskId)
        )
        
        commandEncoder.setComputePipelineState(pipelineKernel)
        commandEncoder.setBuffer(predictionBuffer, offset: 0, index: 0)
        commandEncoder.setBytes(&params, length: MemoryLayout<FilterParams>.stride, index: 1)
        commandEncoder.setBuffer(bboxCount, offset: 0, index: 2)
        commandEncoder.setBuffer(bboxes, offset: 0, index: 3)
        
        commandEncoder.dispatchThreads(
            MTLSize(width: Int(ObjectDetectionModel.stride), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let bboxCountPtr = bboxCount.contents().assumingMemoryBound(to: Int32.self)
        let bboxesPtr = bboxes.contents().assumingMemoryBound(to: MetalBBox.self)
        let count = Int(bboxCountPtr[0])
        var resultMetalBBoxes: [MetalBBox] = []
        
        for i in 0 ..< count {
            resultMetalBBoxes.append(bboxesPtr[i])
        }
        
        var resultBBoxes: [BBox] = []
        for metalBBox in resultMetalBBoxes {
            var box = BBox(classId: metalBBox.classId,
                           confidence: metalBBox.confidence,
                           x: metalBBox.x,
                           y: metalBBox.y,
                           h: metalBBox.h,
                           w: metalBBox.w,
                           numKpts: metalBBox.numKpts,
                           taskId: metalBBox.taskId
            )
            
            let mirror = Mirror(reflecting: metalBBox.kpts)
            for (i, child) in mirror.children.enumerated() {
                if let value = child.value as? Int32 {
                    box.kpts[i] = value
                }
            }
            
            resultBBoxes.append(box)
            
        }
        
        return resultBBoxes
    }
}


class PostProcesserCPU: PostProcessor{
    enum Error: Swift.Error {
        case errorAllocateBuffers
    }
    
    func decodeBox(prediction: MLMultiArray, n_box: UInt32, params: FilterParams) -> BBox? {
        let stride = Int(params.stride)
        let id = Int(n_box)
        
        let start = params.numberOfCoords as UInt32;
        let length = start + params.numberOfClasses as UInt32;
        
        var classId: UInt32 = 0;
        var maxConfidence: Float = 0;
        
        for i in start..<length {
            let confidence = prediction[id + stride * Int(i)].floatValue
            
            if (confidence > maxConfidence) {
                maxConfidence = confidence
                classId = i - start
            }
        }
        
        if (maxConfidence > params.confidenceThreshold) {
            var remainder: Int;
            var value: Float;
            
            let x = prediction[id].floatValue
            let y = prediction[id + stride].floatValue
            let w = prediction[id + stride * 2].floatValue
            let h = prediction[id + stride * 3].floatValue
            
            var box = BBox(
                classId: classId,
                confidence: maxConfidence,
                x: Int32((x - 0.5 * w) * params.factor.x),
                y: Int32((y - 0.5 * h) * params.factor.y),
                h: Int32(h * params.factor.y),
                w: Int32(w * params.factor.x),
                numKpts: params.numberOfKpts,
                taskId: params.taskId
            )
            
            let start = params.numberOfCoords + params.numberOfClasses;
            let end = start + params.numberOfKpts;
            
            for i in start..<end {
                value = prediction[id + stride * Int(i)].floatValue
                remainder = Int(i) % 3
                let kptIndex = Int(i - start)
                
                if (remainder == 0) {
                    box.kpts[kptIndex] = Int32(value * params.factor.y)
                } else if (remainder == 1) {
                    box.kpts[kptIndex] = value > 0.5 ? 1 : 0
                } else if (remainder == 2) {
                    box.kpts[kptIndex] = Int32(value * params.factor.x)
                }
            }
            
            return box
        }
        return nil
    }
    
    func apply(prediction: MLMultiArray, factor: simd_float2 = .one, numberOfClasses: Int = 1, confidenceThreshold: Float = 0.3, numKpts: Int = 0, taskId: Int = 0) throws -> [BBox]? {
        
        let params = FilterParams(
            confidenceThreshold: confidenceThreshold,
            stride: ObjectDetectionModel.stride,
            factor: factor,
            numberOfCoords: UInt32(4),
            numberOfClasses: UInt32(numberOfClasses),
            numberOfKpts: UInt32(numKpts),
            taskId: UInt32(taskId)
        )
        
        // prediction 1, 23, 672
        let shape = prediction.shape as! [Int]
        var bboxes: [BBox] = []
        
        for i in 0..<shape[2] {
            if let bbox = decodeBox(prediction: prediction, n_box: UInt32(i), params: params) {
                bboxes.append(bbox)
            }
        }
        
        return bboxes
    }
}




class PostProcesserCoreML: PostProcesserCPU{
    private let maxHand: MaxModelHand
    private let maxBody: MaxModelBody
    
    override init(){
        maxHand = try! MaxModelHand()
        maxBody = try! MaxModelBody()
    }
    

    override func apply(prediction: MLMultiArray, factor: simd_float2 = .one, numberOfClasses: Int = 1, confidenceThreshold: Float = 0.3, numKpts: Int = 0, taskId: Int = 0) throws -> [BBox]? {
        
        let maxClassPred: MLMultiArray
        if taskId == 0 {
            maxClassPred = try! maxHand.prediction(input: .init(input: prediction)).x1
        } else {
            maxClassPred = try! maxBody.prediction(input: .init(input: prediction)).x1
        }

        let params = FilterParams(
            confidenceThreshold: confidenceThreshold,
            stride: ObjectDetectionModel.stride,
            factor: factor,
            numberOfCoords: UInt32(4),
            numberOfClasses: UInt32(numberOfClasses),
            numberOfKpts: UInt32(numKpts),
            taskId: UInt32(taskId)
        )
        
        // prediction 1, 23, 672
        let shape = prediction.shape as! [Int]
        var bboxes: [BBox] = []
        var count = 0
          
        for i in 0..<shape[2] {
            guard maxClassPred[i].floatValue > confidenceThreshold else { continue }
            count += 1

            if let bbox = decodeBox(prediction: prediction, n_box: UInt32(i), params: params) {
                bboxes.append(bbox)
            }
        }
        
        return bboxes
    }
}
