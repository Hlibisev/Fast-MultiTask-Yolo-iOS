//
//  BanchmarkView.swift
//  Fast-Multitask-Yolov8-IOS
//
//  Created by Anton Fonin on 31.01.2024.
//

import Foundation
import SwiftUI


struct BenchmarkView: View {
    var computedTime: ComputedTime
    
    var body: some View {
        Text("""
             Benchmark
             All: \(Int(1 / computedTime.all)) fps
             Resize: \(Int(1 / computedTime.resize)) fps
             ML model: \(Int(1 / computedTime.model)) fps
             Post proc: \(Int(1 / computedTime.postProc)) fps
             NMS: \(Int(1 / (computedTime.nms + 1e-5))) fps
             """
        )
        .frame(width: 160, height: 160)
        .padding(.all, 15)
        .foregroundStyle(.white)
        .background(.blue)
        .font(.headline)
    }
}


#Preview {
    BenchmarkView(computedTime: ComputedTime())
}

