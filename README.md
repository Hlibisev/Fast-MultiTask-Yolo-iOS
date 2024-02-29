# Fast-MultiTask-Yolo-iOS
Efficient Algorithm for Gesture Recognition and Body Keypoints Detection. 

https://github.com/Hlibisev/Fast-MultiTask-Yolo-iOS/assets/64321152/aa0694cb-b35a-42d6-88b6-0df215b48340

## Introduction
There are primarily two tools for efficiently running neural networks on Apple devices. 

* Metal and Metal Shader Language (MSL). Neural networks operate effectively on GPUs. MSL allows for writing highly efficient low-level code, while the Metal API facilitates its execution.
* The CoreML framework, which takes a static graph, for example, after JIT torch, and converts it into a format for running on Apple devices without the need for a Python interpreter or Torch. It maps operations from this graph to operations written by Apple for the GPU (using Metal), NPU, or CPU.

NPU – a more efficient chip for matrix multiplication calculations. Direct programming on it is not possible, but it can be utilized for certain layers when we convert our model to CoreML format. More information about NPU can be found here.

Using these two tools, as well as standard CPU code, I aimed to write an optimal implementation of the MultiTask Yolov8 algorithm on the iPhone.


## NPU vs GPU: Performance Analysis

Initially, I tested the frames per second (fps) performance on models converted to CoreML format.

shape (224, 320)
- **CPU**: 93 fps
- **GPU**: 61 fps
- **NPU**: 370 fps

shape (128, 256)
- **CPU**: 132 fps
- **GPU**: 82 fps
- **NPU**: 550 fps

We observe that the NPU chip has the highest performance. It's sufficient for the current stage, and I decided to focus on the preprocessing and postprocessing steps.

## Preproceeing
For our model, preprocessing outside the CoreML Model only included resizing.

### Baseline
As a test, I initially performed resizing on the CPU but realized it took much longer than the model itself. Thus, I quickly moved to MPSImageBilinearScale. This GPU-based resizing operation operates at approximately **600 fps**. This will be our baseline, which we will aim to improve further.

All attempts were inspired by the idea of transferring resizing to the NPU or reducing the overhead of launching GPU operations by incorporating this operation into a static graph.

### Resize Inside Layer
Since direct access to the NPU is not available – thanks, Apple – I began experimenting with writing a version of resize in PyTorch, which, after conversion to CoreML, would utilize the NPU. 

```python
class ResizeLayer(torch.nn.Module):
    def __init__(self):
        super(ResizeLayer, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(320, 224), mode="bilinear")

        return x
```

However, due to an internal error in 'coremltools' (the library for converting Torch models into CoreML format, which can run on phones and MacBooks), I couldn't compile this model. But later, with operations implemented by Apple, I redefined upsample_bilinear2d and managed to get a working model, though it only reached **270 fps** compared to our baseline. Therefore, I tried another approach.

```python
@register_torch_op(torch_alias=["upsample_bilinear2d"], override=True)
def resize(context, node):
    x, shape, _, _ = _get_inputs(context, node, expected=4)

    x = mb.resize_bilinear(x=x, target_size_height=shape.val[0], target_size_width=shape.val[1], name=node.name)
    context.add(x)
```


### Nearest Torch Realization Layer
Next, I decided not to use Apple operations but to write my resizer using regular torch functions. Such a nearest resizer was also worse than the baseline and reached **170 fps** after conversion.

```python
def upsample_1d(x, shape1, axis: int = 2):
    shape0 = x.size(axis)

    X = torch.arange(shape0)
    Y = torch.arange(shape1)

    _X = X.float() * shape1 / shape0
    _Y = Y.float()

    C = (-(_X.unsqueeze(0) - _Y.unsqueeze(1)).abs()).argmax(dim=1).long()

    x = torch.index_select(x, axis, X[C])
    return x

@torch.jit.script_if_tracing
def upsample_2d(x, shapes):
    x = upsample_1d(x, shapes[0], axis=2)
    x = upsample_1d(x, shapes[1], axis=3)
    return x
```

### Resize with Convolution

The input size of the frontal camera on iPhone 13 is 1920, 1080. The idea was to translate interpolation to a convolution layer. This may be more efficient and could run on NPU instead of GPU.


I wrote a convolution that resembles resizing. Although it just reduces the input image by 16 times, and you cannot choose another resolution, I still wanted to test this hypothesis.

```python
class ResizeConv(torch.nn.Module):
    def __init__(self):
        super(ResizeConv, self).__init__()
        self.resize_conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1, stride=6, groups=3)

        self.resize_conv.weight.data.fill_(1 / 25)
        self.resize_conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.resize_conv(x)
        return x
```

Running on a MacBook or iPhone with a hardcoded shape, it has **130 fps** on NPU, **100 fps** on GPU instead of **600 fps** metal shader.

However, here I learned the following feature. CoreML cannot include NPU if the model has dynamic input (even if part of the mlmodel is executed at the same resolution, as after resizing). This greatly complicates the situation in general when we do not know what resolution will be at the input. We will explore this approach later at 1920 by 1080.


### Custom Resize Inside CoreML
What if the setup time for the resize kernel is too long? Yes, that's the case. Therefore, I decided to try adding my neural layer, implemented in Metal, to the computational graph to remove overheads and have a static graph. However, it turned out that neural networks with custom layers do not support NPU at all. This could be a very strong limitation for porting nn to mobile devices.

Therefore, I also discarded this option. If the neural network worked on the GPU as well or faster than on the NN, we could consider such an option.



### Nearest Resize with Metal Kernel
Next, I tried to implement my nearest resizing kernel to compare its speed with Apple's. Billinear generally works better than Nearest, but here we win about **50-100 fps**.

```msl
kernel void nearestResize(texture2d<float, access::sample> source [[ texture(0) ]],
                          texture2d<float, access::write> destination [[ texture(1) ]],
                          uint2 gid [[ thread_position_in_grid ]])
{
    if (gid.x >= destination.get_width() || gid.y >= destination.get_height()) {
        return;
    }

    float2 scale = float2(source.get_width(), source.get_height()) / float2(destination.get_width(), destination.get_height());

    float2 srcCoords = (float2(gid) + 0.5f) * scale - 0.5f;
    float4 color = source.read(uint2(srcCoords));

    destination.write(color, gid);
}
```

### Pipeline model
As an experiment, I had a hypothesis to connect resize and yolo into one model with hardcoded resolution, and this can be done through coremltools.

```python
resizer = ct.convert(
    traced_resize_model,
    inputs=[ct.ImageType(name="input", shape=(1, 3, 1920, 1080))],
    outputs=[ct.ImageType(name="image")],
    convert_to="mlprogram",
)

model = ct.models.MLModel("Yolo2_320_224.mlpackage")

pipeline = ct.models.pipeline.Pipeline(
    input_features=[('input', ct.models.datatypes.Array(1, 3, 1920, 1080))],
    output_features=['var_897', 'var_1335']
)

pipeline.add_model(resizer)
pipeline.add_model(model)

pipeline.spec.description.input[0].ParseFromString(resizer._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(model._spec.description.output[1].SerializeToString())


model = ct.models.MLModel(pipeline.spec, weights_dir=model.weights_dir)
``` 
I tried all torch resizes, and they yielded **100-130 fps** with the model. This is less than the baseline resizer + CoreML model but more than without combining them into a pipeline.

### One static model

```python
class IOSMultiModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resize = ResizeConv()

    def forward(self, x):
        x = self.resize(x)
        detection, pose = self.model(x)
        return detection, pose
```

My last attempt to speed up resizing – into one mlmodel, and so far, this is the best attempt without using Metal – **185 fps**. It remains about **35 fps** compared to Metal resize + CoreML.

I achieved this result using torchvision.

```python
class UpsampleModel(torch.nn.Module):
    def __init__(self):
        super(UpsampleModel, self).__init__()

    def forward(self, x):
        x = F.resize(x, (320, 224), interpolation=F.InterpolationMode.NEAREST)

        return x
```

I stop searching at Metal Resize + NPU CoreML. I could not find how to perform resizing on NPU, adding my custom NN Resizing layer is not possible in CoreML without switching from NPU to GPU, and the combined compiled model does not offer an advantage over the basic version.


## Post Processing
This part is about creating a response from the neural network tensor in a format convenient for Swift. I had 3 attempts to make post-processing. GPU, CPU, CPU + CoreML operations. Here I won't elaborate much, just to say,

GPU worked most efficiently of all and since I need to call it for two outputs, I do it simultaneously asynchronously. This leads me to **500-600 fps**.

Afterward, I apply the Non-Max Suppression algorithm written on CPU with 26k fps.

## Conclusion
In the end, the entire model reaches **153 fps** at a resolution of 320 by 224, and **197 fps** at 256 by 128.
