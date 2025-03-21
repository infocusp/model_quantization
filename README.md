# model_quantization
Tutorial / seminar on quantization of DL models.

# What is Quantization?

- Quantization is essential for reducing the size and computational requirements of AI models, especially for deployment on edge devices with limited resources.
- Large neural networks require high memory and processing power, which makes them unsuitable for real-time inference on edge devices like smartphones, IoT devices, and embedded systems.
- Quantization involves representing weights and activations of a neural network using lower precision (e.g., 8-bit integers instead of 32-bit floating-point numbers).

## Pros
- This conversion of data type to integer can reduce memory footprint by upto 4x.
- It speedsup inference, as integer operations are faster than floating point operations thus reducing compute time approximately.

## Cons
- Due to quantization, we expect some loss in accuracy as it introduces approximation in the model weights.
- Not all hardware supports all quantization formats (e.g., INT8 vs. FP16).


# Symetric vs Assymetric Quantization.
- There are two ways to convert the FP values to integers.


## Symmetric quantization
- The range of floating-point values is mapped symmetrically around zero.
- The same scale factor is used for both positive and negative values.
- Simple, efficient, works best if data is evenly distributed around zero.
- Attached notebook for reference.


## Asymmetric quantization
- The range of floating-point values is mapped using different scales for positive and negative values.
- More precise, better for real-world data that isn’t evenly spread.
- Attached notebook for reference.



# How a floating point is represented in 32 bit binary format.
- Do show show floating point numer is represent in binary format, we'll take an example to convert a floating point number 85.125 into 32 bit binary.

- Step 1: Convert 85 to binary : 1010101
- Step 2: Convert 0.125 to binary: 001
- Step 3: Combine both the part: 85.125 -> 1010101.001
- Step 4: We will shift the . to the second positing and multiple by 2*n where n is the number of digit shifted. Therefore 85.125 -> 1010101.001 -> 1.010101001 x 2 ^ 6
- Step 5: Add 127 to the exponent and get binary of the resultant number: 127 + 6 = 133 = 10000101
- Step 6:
    - Now we have everything. The first bit of 32 bit represent sign of the number in our case the number is positive therefore it should be 0.
    - The next 8 bit represent exponent so we have binary of 133 that is 10000101
    - The remaning value will be the value after . in step 5. That is 010101001 which is also called as mentassa. We place 0 in the padding.
- Step 7: Therefore 85.125 in 32 bit format will be 0 10000101 010101001 00000000000000



# How to achieve Quantization?
Quantization can be achieved by following primary techniques.

## 1. Post training quantization (PTQ)
- In post training quantization, we quantize the model after actual training is done, i.e we quantize the pretrained model.
- We make a copy of actual architecture and add quantize layer to start and dequantize layer to last.
- We also add observer in between which help in understanding stats about min-max values in each layer.
- We run inference on this quantized architecture, and calculate the parameters like scale, zero-point that helps to quantize the model.
- Observation is that the size of model reduces upto 4x, as FP32 is converted to INT8.
- Also, speedup is observed as in general computations are faster on integers than floating point values.
- Some degradation in performance is expected due to low precision conversion.
- There are two types for this quantization:
    - a) Static
    - b) Dynamic
- More details in tutorial notebooks.


## 3. Quantization Aware training (QAT)
- In thi approach, we try to train model with quantization such that it adjusts better with the loss functions.
- We quantize and deqauntize between every layer of the network.
- For back propogation, we approximate the gradient since this quantization is non-differentiable.
- QAT is more robust compared to PTQ, as it learns better during actual model training aiding the losses to converge accordingly.


## Tutorial Notebooks

| Topic Name    | PyTorch Tutorials |
| -------- | ------- |
|Post Training Dynamic Quantization|  [PTDQ_Tutorial.ipynb](#link) |
|Post Training Static Quantization| [PTSQ_Tutorial.ipynb](#link) |
|Quantization Aware Training|[QAT_Tutorial.ipynb](#link)|
|Quantization Aware Training|[QAT_Tutorial.ipynb](#link)|
|Quantization using Bits and Bytes|[bitsandbytes.ipynb](#link)|


# References

**Notebooks**
- [Tensorflow Post training quantization](https://colab.research.google.com/drive/16FnHAiGEZREZ_FYU7Zs6eGQ2CBrklwdG)
- [Tensorflow Quantization aware training](https://colab.research.google.com/drive/19bP6iSfaXSJinxNzlBqr6sWCTW0oki4l)

**Articles**
- [How to Convert a Number from Decimal to IEEE 754 Floating Point Representation](https://www.wikihow.com/Convert-a-Number-from-Decimal-to-IEEE-754-Floating-Point-Representation)
- [Tensorflow Post training Quantization](https://ai.google.dev/edge/litert/models/post_training_quantization)
- [Tensorflow Quantization aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)


**Video**
- [Quantization explained with PyTorch - Post-Training Quantization, Quantization-Aware Training
](https://youtu.be/0VdNflU08yA?si=ITIOJvxSRApH0wzj)

**Presentation**
# link of ppt

