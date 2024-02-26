import Foundation
import AVFoundation
import MetalKit


typealias PixelFormat = OSType

extension CVPixelBufferPool {
    static func allocate(
        for dimension: CMVideoDimensions,
        pixelFormat: PixelFormat,
        bufferSize: UInt32
    ) -> CVPixelBufferPool? {
        let outputBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
            kCVPixelBufferWidthKey as String: Int(dimension.width),
            kCVPixelBufferHeightKey as String: Int(dimension.height),
            kCVPixelBufferIOSurfacePropertiesKey as String: [:]
        ]
        let poolAttributes = [kCVPixelBufferPoolMinimumBufferCountKey as String: bufferSize]
        var pixelBufferPool: CVPixelBufferPool?
        let result = CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as NSDictionary?,
            outputBufferAttributes as NSDictionary?,
            &pixelBufferPool
        )
        guard result == kCVReturnSuccess, let pixelBufferPool = pixelBufferPool else {
            return nil
        }
        return pixelBufferPool
    }
}

extension CVPixelBuffer {
    var isPlanar: Bool {
        CVPixelBufferIsPlanar(self)
    }
    
    var mtlSize: MTLSize {
        MTLSize(
            width: CVPixelBufferGetWidth(self),
            height: CVPixelBufferGetHeight(self),
            depth: 1
        )
    }
    
    var size: CGSize {
        CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
    }
    
    var bytesPerRow: Int {
        CVPixelBufferGetBytesPerRow(self)
    }
    
    var bytesCount: Int {
        let height = CVPixelBufferGetHeight(self)
        return height * bytesPerRow
    }
    
    func getSize(forPlane planeIndex: Int) -> CGSize {
        CGSize(
            width: CVPixelBufferGetWidthOfPlane(self, planeIndex),
            height: CVPixelBufferGetHeightOfPlane(self, planeIndex)
        )
    }
    
    func getMTLSize(forPlane planeIndex: Int) -> MTLSize {
        MTLSize(
            width: CVPixelBufferGetWidthOfPlane(self, planeIndex),
            height: CVPixelBufferGetHeightOfPlane(self, planeIndex),
            depth: 1
        )
    }
    
    func getBytesPerRow(forPlane planeIndex: Int) -> Int {
        CVPixelBufferGetBytesPerRowOfPlane(self, planeIndex)
    }
    
    func getBytesCount(forPlane planeIndex: Int) -> Int {
        let height = CVPixelBufferGetHeightOfPlane(self, planeIndex)
        return height * getBytesPerRow(forPlane: planeIndex)
    }
    
    func makeMTLTexture(
        usingTextureCache textureCache: CVMetalTextureCache,
        pixelFormat: MTLPixelFormat,
        planeIndex: Int = 0
    ) -> MTLTexture? {
        if let cvMetalTexture = CVMetalTexture.createFromCVPixelBuffer(
            self, usingTextureCache: textureCache, pixelFormat: pixelFormat, planeIndex: planeIndex
        ) {
            return CVMetalTextureGetTexture(cvMetalTexture)
        } else {
            return nil
        }
    }
}
