import AVFoundation
import CoreImage
import SwiftUI


protocol PartFrameDetegator: AnyObject {
    func cameraController(didOutput pixelBuffer: CVPixelBuffer) -> [BBox]?
}


class FrameDataHandler: NSObject, ObservableObject {
    @Published var frame: CGImage?
    @Published var points: [Circle] = []
    @Published var isVisible: [Bool] = []
    @Published var positions: [CGPoint] = []
    @Published var colors: [Color] = []
    @Published var bboxes: [BBox] = []
    @Published var computedTime = ComputedTime()
    
    private var permissionGranted = true
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "sessionQueue")
    private let context = CIContext()
    private let maxNumPoints: Int
    
    private let delegate = ObjectDetector(postDevice: .GPU)
    private var framesAmount = 0
    
    init(maxNumPoints: Int = 1000) {
        self.maxNumPoints = maxNumPoints
        
        super.init()
        
        self.checkPermission()
        
        points = Array(repeating: Circle(), count: maxNumPoints)
        isVisible = Array(repeating: false, count: maxNumPoints)
        positions = Array(repeating: .zero, count: maxNumPoints)
        colors = Array(repeating: .black, count: maxNumPoints)
        
        sessionQueue.async { [unowned self] in
            self.setupCaptureSession()
            self.captureSession.startRunning()
        }
    }
    
    func checkPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            self.permissionGranted = true
            
        case .notDetermined:
            self.requestPermission()
            
        default:
            self.permissionGranted = false
        }
        print("permisstion", self.permissionGranted)
    }
    
    func requestPermission() {
        AVCaptureDevice.requestAccess(for: .video) { [unowned self] granted in
            self.permissionGranted = granted
        }
    }
    
    func setupCaptureSession() {
        let videoOutput = AVCaptureVideoDataOutput()
        
        guard permissionGranted else { return }
        guard let videoDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else { return }
        
        //        guard let videoDevice = AVCaptureDevice.default(for: .video) else { return }
        
        guard let videoDeviceInput = try? AVCaptureDeviceInput(device: videoDevice) else { return }
        guard captureSession.canAddInput(videoDeviceInput) else { return }
        captureSession.addInput(videoDeviceInput)
        
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "sampleBufferQueue"))
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA),
        ]
        
        captureSession.addOutput(videoOutput)
        videoOutput.connection(with: .video)?.videoOrientation = .portrait
        
        print("setupCaptureSession")
    }
}

extension FrameDataHandler {
    func setPredictionData(bboxes: [BBox]) {
        self.bboxes = bboxes
        
        isVisible = Array(repeating: false, count: maxNumPoints)
        
        var index = 0
        
        for box in bboxes {
            if box.numKpts > 0 {
                setKpts(box: box, index: &index)
            }
        }
    }
    
    func setKpts(box: BBox, index: inout Int){
        var array: [Int32] = []
        
        let mirror = Mirror(reflecting: box.kpts)
        for child in mirror.children {
            if let value = child.value as? Int32 {
                array.append(value)
            }
        }
        
        for i in stride(from: 0, to: Int(box.numKpts), by: 3) {
            guard index < maxNumPoints else { return }
            guard array[i + 2] == 1 else { continue }
            
            positions[index] = CGPoint(x: Double(array[i]), y: Double(array[i + 1]))
            isVisible[index] = true
            colors[index] = .mint
            
            index += 1
        }
    }
    
    func setTimeInfo(computedTime: ComputedTime) {
        guard framesAmount % 15 == 0 else { return }
        self.computedTime = computedTime
    }
}


extension FrameDataHandler: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        framesAmount += 1
        
        let bboxes = delegate.cameraController(didOutput: imageBuffer) ?? []
        
        DispatchQueue.main.async { [unowned self] in
            self.frame = imageFromSampleBuffer(sampleBuffer: sampleBuffer)
            self.setPredictionData(bboxes: bboxes)
            self.setTimeInfo(computedTime: delegate.computedTime)
        }
    }
    
    private func imageFromSampleBuffer(sampleBuffer: CMSampleBuffer) -> CGImage? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ciImage = CIImage(cvPixelBuffer: imageBuffer)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        
        return cgImage
    }
    
}
