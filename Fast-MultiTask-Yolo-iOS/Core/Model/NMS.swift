//
//  NMS.swift
//  postprocessing_yolo
//
//  Created by Anton Fonin on 15.12.2023.
//

import Foundation


func IOU(box1: BBox, box2: BBox) -> Float {
    let x1 = max(box1.x, box2.x)
    let y1 = max(box1.y, box2.y)
    let x2 = min(box1.x + box1.w, box2.x + box2.w)
    let y2 = min(box1.y + box1.h, box2.y + box2.h)
    let w = max(y2 - y1, 0)
    let h = max(x2 - x1, 0)
    let intersection = w * h;
    let a1 = box1.w * box1.h;
    let a2 = box2.w * box2.h;
    return Float(intersection) / Float(a1 + a2 - intersection);
}


func applyNMS(bboxes: [BBox], IOU_max: Float) -> [BBox] {
    var index: Int?
    var bboxes = bboxes
    var output: [BBox] = []
    
    while !bboxes.isEmpty {
        index = bboxes.map { $0.confidence }.argmax()
        guard let index else { return output }
        let box = bboxes[index]
        
        output.append(box)
        
        bboxes = bboxes.filter { IOU(box1: $0, box2: box) < IOU_max }
    }
    return output
}


extension Array where Element: Comparable {
    func argmax() -> Int? {
        return self.enumerated().max(by: { $0.element < $1.element })?.offset
    }
}
