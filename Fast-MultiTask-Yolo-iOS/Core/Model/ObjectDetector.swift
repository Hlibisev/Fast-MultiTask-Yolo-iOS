//
//  ObjectDetector.swift
//  postprocessing_yolo
//
//  Created by Anton Fonin on 15.12.2023.
//

import Foundation
import CoreML
import AVFoundation
import VideoToolbox
import CoreImage
import SwiftUI


struct ComputedTime {
    var resize: Double = 1
    var model: Double = 1
    var postProc: Double = 1
    var nms: Double = 1
    
    var all: Double {
        get {
            resize + model + postProc + nms
        }
    }
    
    mutating func update(new: ComputedTime, coef: Double = 0.1) {
        resize = new.resize * coef + resize * (1 - coef)
        model = new.model * coef + model * (1 - coef)
        postProc = new.postProc * coef + postProc * (1 - coef)
        nms = new.nms * coef + nms * (1 - coef)
    }
}



class ObjectDetector: NSObject{
    private var model: ObjectDetectionModel
    private var postProcesser: PostProcessor
    private let resizer: MTLImageScaler
    
    public var computedTime = ComputedTime()

    enum postProcessType {
        case CPU
        case GPU
        case CoreML
    }
    
    init(postDevice: postProcessType = .CPU) {
        model = ObjectDetectionModel()
        resizer = MTLImageScaler(rescaledSize: ObjectDetectionModel.inputSize)
        
        switch postDevice {
        case .CPU:
            postProcesser = PostProcesserCPU()
        case .GPU:
            postProcesser = PostProcesserGPU()
        case .CoreML:
            postProcesser = PostProcesserCoreML()
        }
        
        super.init()
        try! model.load()
    }
    
    func parallelPostprocess(data: ObjectDetectionModel.Output, factor: simd_float2) -> ([BBox], [BBox]){
        var bboxesHand: [BBox] = []
        var bboxesBody: [BBox] = []
        
        let group = DispatchGroup()
        
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            bboxesHand = try! self.postProcesser.apply(
                prediction: data.hand,
                factor: factor,
                numberOfClasses: ObjectDetectionModel.classes[0, default: [:]].count,
                confidenceThreshold: 0.3,
                numKpts: 0,
                taskId: 0
            ) ?? []
            group.leave()
        }
        
        group.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            bboxesBody = try! self.postProcesser.apply(
                prediction: data.body,
                factor: factor,
                numberOfClasses: ObjectDetectionModel.classes[1, default: [:]].count,
                confidenceThreshold: 0.5,
                numKpts: 51,
                taskId: 1
            ) ?? []
            group.leave()
        }
        
        group.wait()
        
        return (bboxesHand, bboxesBody)
    }
    
    func apply(imPixelBuffer: CVPixelBuffer, factor: simd_float2 = .one) -> [BBox]? {
        var thisFrameComputedTime = ComputedTime()
                
        var startTime = Date()
        let resizedBuffer = try! resizer.rescale(imPixelBuffer)
        thisFrameComputedTime.resize = Date().timeIntervalSince(startTime)
        
        startTime = Date()
        guard let output = model.predict(image: resizedBuffer) else { return nil }
        thisFrameComputedTime.model = Date().timeIntervalSince(startTime)
        
        startTime = Date()
        var (bboxesHand, bboxesBody) = parallelPostprocess(data: output, factor: factor)
        thisFrameComputedTime.postProc = Date().timeIntervalSince(startTime)
        
        startTime = Date()
        bboxesHand = applyNMS(bboxes: bboxesHand, IOU_max: 0.4)
        bboxesBody = applyNMS(bboxes: bboxesBody, IOU_max: 0.3)
        thisFrameComputedTime.nms = Date().timeIntervalSince(startTime)

        computedTime.update(new: thisFrameComputedTime)
        return bboxesHand + bboxesBody
    }
}


extension ObjectDetector: PartFrameDetegator {
    func cameraController(didOutput pixelBuffer: CVPixelBuffer) -> [BBox]? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let factor = simd_float2(
            x: Float(ciImage.extent.width) / Float(ObjectDetectionModel.inputSize.width),
            y: Float(ciImage.extent.height) / Float(ObjectDetectionModel.inputSize.height)
        )
        
        let bboxes = self.apply(imPixelBuffer: pixelBuffer, factor: factor)
        return bboxes
    }
}

