//
//  ContentView.swift
//  postprocessing_yolo
//
//  Created by Anton Fonin on 14.12.2023.
//

import SwiftUI
import CoreML
import CoreGraphics


struct ContentView: View {
    @StateObject private var frameDataHandler = FrameDataHandler()
    
    var body: some View {
        let image = frameDataHandler.frame
        let scaleHeight = CGFloat(image?.height ?? 1) / UIScreen.main.bounds.height
        let scaleWidth = CGFloat(image?.width ?? 1) / UIScreen.main.bounds.width
        
        ZStack {
            ZStack {
                CameraView(image: frameDataHandler.frame)
                PointsView(
                    points: frameDataHandler.points,
                    isVisible: frameDataHandler.isVisible,
                    positions: frameDataHandler.positions,
                    colors: frameDataHandler.colors
                )
                BoundingBoxView(boundingBox: frameDataHandler.bboxes)
            }
            .scaleEffect(min(1 / scaleHeight, 1 / scaleWidth, 1))
            .frame(width: UIScreen.main.bounds.width, height: UIScreen.main.bounds.height)
            .clipped()
            
            
            VStack {
                HStack {
                    Spacer()
                    BenchmarkView(computedTime: frameDataHandler.computedTime)
                }
                .padding(.horizontal, 20)
                .padding(.top, 50)
                Spacer()
            }
        }
    }
}


#Preview {
    ContentView()
}

