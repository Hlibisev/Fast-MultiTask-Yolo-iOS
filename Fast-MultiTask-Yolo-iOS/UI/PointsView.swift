import SwiftUI


struct PointsView: View {
    var points: [Circle] = []
    var isVisible: [Bool]
    var positions: [CGPoint]
    var colors: [Color]
    
    var body: some View {
        ZStack {
            ForEach(0..<100, id: \.self) { index in
                if isVisible[index] {
                    points[index]
                        .fill(colors[index])
                        .frame(width: 25, height: 25)
                        .position(positions[index])
                }
            }
        }
    }
}

