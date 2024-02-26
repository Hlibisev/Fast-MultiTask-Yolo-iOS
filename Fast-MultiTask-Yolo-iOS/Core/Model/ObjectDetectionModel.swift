
//
// Created by moonl1ght 27.02.2023.
//

import Foundation
import CoreML

final class ObjectDetectionModel {
    enum Error: Swift.Error {
        case failedToLoadModel
    }
    
    static let classes: [UInt32: [UInt32: String]] = [
        0: [
            0: "Call",
            1: "Dislike",
            2: "Fist",
            3: "Four",
            4: "Like",
            5: "Mute",
            6: "Ok",
            7: "One",
            8: "Palm",
            9: "Peace",
            10: "Rock",
            11: "Stop",
            12: "Stop inverted",
            13: "Three",
            14: "Two up",
            15: "Two up inverted",
            16: "Three2",
            17: "Peace inverted",
            18: "No gesture",
        ],
        1: [0: "Person"]
    
    ]
    
    final class Output {
        let hand: MLMultiArray
        let body: MLMultiArray
        
        init(hand: MLMultiArray, body: MLMultiArray) {
            self.hand = hand
            self.body = body
        }
    }
    
    static let inputSize = CGSize(width: 224, height: 320)
    static let stride: Int32 = 1470
    
    private var modelMulti: Yolo2_320_224?
    
    func load() throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        modelMulti = try Yolo2_320_224(configuration: configuration)

    }

    func predict(image: CVPixelBuffer) -> Output? {
        do {
            guard let result = try modelMulti?.prediction(image: image) else { return nil }
            return Output(hand: result.var_897, body: result.var_1335)
        } catch {
            assertionFailure(error.localizedDescription)
            return nil
        }
    }
}
