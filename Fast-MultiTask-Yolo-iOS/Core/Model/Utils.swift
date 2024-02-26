//
//  Utils.swift
//  postprocessing_yolo
//
//  Created by Anton Fonin on 14.12.2023.
//

import Foundation
import MetalKit
import CoreML



enum MTLPipelineState {
    static func createCompute(
        library: MTLLibrary,
        device: MTLDevice,
        functionName: String,
        label: String? = nil
    ) -> MTLComputePipelineState {
        do {
            let computePipelineDescriptor = MTLComputePipelineDescriptor()
            let converterFunction = library.makeFunction(name: functionName)
            computePipelineDescriptor.computeFunction = converterFunction
            if let label {
                computePipelineDescriptor.label = label
            }
            computePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
            return try device.makeComputePipelineState(
                descriptor: computePipelineDescriptor, options: [], reflection: nil
            )
        } catch {
            fatalError(error.localizedDescription)
        }
    }
}

extension MLMultiArray {
    func toMTLBuffer(device: MTLDevice) -> MTLBuffer? {
        let predictionBuffer: MTLBuffer? = self.withUnsafeMutableBytes { ptr, strides in
            guard let fptr = ptr.assumingMemoryBound(to: Float.self).baseAddress else {
                return nil
            }
            return device.makeBuffer(
                bytes: fptr,
                length: MemoryLayout<Float>.stride * self.count,
                options: [.storageModeShared]
            )
        }
        
        return predictionBuffer
    }
}


class BufferManager {
    private var bufferCache: [Int: MTLBuffer] = [:]
    private let device: MTLDevice
    private let lock = NSLock()
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    func buffer(for multiArray: MLMultiArray) -> MTLBuffer? {
        // wo lock function doesn't work in different thread
        lock.lock()
        defer { lock.unlock() }
        
        let length = MemoryLayout<Float>.stride * multiArray.count
        
        if let buffer = bufferCache[length] {
            buffer.contents().copyMemory(from: multiArray.dataPointer, byteCount: length)
            return buffer
        }
        
        guard let newBuffer = device.makeBuffer(length: length, options: []) else {
            return nil
        }
        
        bufferCache[length] = newBuffer
        
        return newBuffer
    }
}
