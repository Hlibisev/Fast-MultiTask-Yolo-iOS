//
//  BBoxView.swift
//  Fast-Multitask-Yolov8-IOS
//
//  Created by Anton Fonin on 09.01.2024.
//

import Foundation
import SwiftUI


struct BoundingBoxView: View {
    var boundingBox: [BBox]
    
    var body: some View {
        ForEach(boundingBox, id: \.classId) { box in
            let centerX = CGFloat(box.x) + CGFloat(box.w) / 2.0
            let centerY = CGFloat(box.y) + CGFloat(box.h) / 2.0
            let taskClasses = ObjectDetectionModel.classes[box.taskId, default: [:]]
            let className = taskClasses[box.classId, default: "Unknown"]
            
            Rectangle()
                .stroke(Color.red, lineWidth: 4.0)
                .frame(width: CGFloat(box.w), height: CGFloat(box.h))
                .position(x: CGFloat(centerX), y: CGFloat(centerY))
            
            Text(className)
                .background(Color.red)
                .foregroundColor(.white)
                .font(.system(size: 80))
                .padding(10)
                .position(x: centerX, y: centerY - CGFloat(box.h) / 2.0 - 80)
        }
    }
}
