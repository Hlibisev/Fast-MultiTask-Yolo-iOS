import Foundation
import MetalKit

extension CVMetalTexture {
    static func createFromCVPixelBuffer(
        _ cvPixelBuffer: CVPixelBuffer,
        usingTextureCache textureCache: CVMetalTextureCache,
        pixelFormat: MTLPixelFormat,
        planeIndex: Int = 0
    ) -> CVMetalTexture? {
        precondition(planeIndex >= 0, "Plane index must be non negative.")
        let size: MTLSize
        if cvPixelBuffer.isPlanar {
            size = cvPixelBuffer.getMTLSize(forPlane: planeIndex)
        } else {
            size = cvPixelBuffer.mtlSize
        }
        var texture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, cvPixelBuffer, nil, pixelFormat, size.width, size.height, planeIndex, &texture
        )
        if status != kCVReturnSuccess {
            texture = nil
            assertionFailure("Failed to create texture from CVPixelBuffer.")
        }
        return texture
    }
}


extension CVMetalTextureCache {
    static func createUsingDevice(_ device: MTLDevice) -> CVMetalTextureCache {
        var textureCache: CVMetalTextureCache?
        let result = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)
        if result == kCVReturnSuccess, let textureCache {
            return textureCache
        } else {
            fatalError("Failed to create CVMetalTextureCache")
        }
    }
}
