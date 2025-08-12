# Evolution of CNN Architectures for ImageNet

This README traces the development of notable convolutional neural network (CNN) architectures that have competed in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC). Chronologically ordered, each section covers:

- Brief description  
- Year of ImageNet result  
- Research paper (hyperlinked)  
- Key novelty  
- Advantages  
- Disadvantages  
- Problems it addressed compared to its predecessors

---

## 1. AlexNet (~61 M parameters)
- **Brief description**  
  A breakthrough deep CNN with 5 convolutional layers and 3 fully connected layers, using ReLU activations, dropout, and GPU-optimized training.

- **Year of ImageNet result**  
  2012

- **Research Paper**  
  [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

- **Novelty**  
  Introduced large-scale use of ReLU, dropout for regularization, and data augmentation.

- **Advantages**  
  Greatly reduced error rates, showcased feasibility of very deep CNNs trained on GPUs.

- **Disadvantages**  
  Huge parameter count (~61M), risk of overfitting, resource-heavy.

- **Problems solved**  
  Demonstrated that deeper models could outperform shallow ones using GPU acceleration and regularization, pioneering modern CNN training.

---

## 2. GoogLeNet (Inception v1) (~6.8 M parameters)
- **Brief description**  
  Introduced the Inception module—a multi-branch architecture combining convolutions and pooling in parallel.

- **Year of ImageNet result**  
  2014

- **Research Paper**  
  [Going Deeper with Convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)

- **Novelty**  
  Inception module: multiple filter sizes in parallel within one layer for multi-scale feature capture.

- **Advantages**  
  Much fewer parameters (~6.8M), efficient computation, deeper network with improved accuracy.

- **Disadvantages**  
  Complex architecture; harder to design and tune manually.

- **Problems solved**  
  Reduced model size drastically compared to AlexNet, improved efficiency while increasing depth.

---

## 3. VGG16 (~138 M parameters)
- **Brief description**  
  Extremely deep (16 layers), using only 3×3 convolution filters and fixed network uniformity.

- **Year of ImageNet result**  
  2014

- **Research Paper**  
  [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

- **Novelty**  
  Uniform architecture using small, stacked convolutions to increase depth.

- **Advantages**  
  Straightforward to implement, effective pre-trained model for transfer learning.

- **Disadvantages**  
  Very large parameter count (~138M), high computational cost and memory usage.

- **Problems solved**  
  Showed that increasing depth via small kernels improves performance, though at a computational cost.

---

## 4. ResNet-50 (~25.6 M parameters)
- **Brief description**  
  Introduced residual learning via skip connections to enable training of very deep networks.

- **Year of ImageNet result**  
  2015

- **Research Paper**  
  [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

- **Novelty**  
  Residual (skip) connections that ease gradient flow in deep networks.

- **Advantages**  
  Enables very deep models that are easier to train; improved accuracy with moderate parameter size (~25.6M).

- **Disadvantages**  
  Deeper networks still require significant computation and memory.

- **Problems solved**  
  Addressed vanishing gradients by facilitating learning of deeper architectures more reliably.

---

## 5. DenseNet-121 (~8 M parameters)
- **Brief description**  
  Dense connectivity: each layer receives inputs from all preceding layers.

- **Year of ImageNet result**  
  2016

- **Research Paper**  
  [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

- **Novelty**  
  Dense skip connections; encourages feature reuse throughout the network.

- **Advantages**  
  Parameter-efficient (~8M), improved information and gradient flow, strong feature propagation.

- **Disadvantages**  
  Computation-heavy, memory usage can grow due to concatenation of many features.

- **Problems solved**  
  Enhanced gradient flow and feature reuse more effectively than ResNet, while using fewer parameters.

---

## 6. SqueezeNet (~1.2 M parameters)
- **Brief description**  
  Compact network using “fire” modules—squeeze (1×1) then expand (1×1 & 3×3) convolutions.

- **Year of ImageNet result**  
  2016

- **Research Paper**  
  [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)

- **Novelty**  
  Fire modules and aggressive bottlenecking for maximal parameter reduction.

- **Advantages**  
  Tiny model (~1.2M parameters), great for deployment on resource-constrained devices.

- **Disadvantages**  
  Slightly lower accuracy compared to larger models.

- **Problems solved**  
  Showed competitive accuracy achievable with very small models—beneficial for embedded systems.

---

## 7. MobileNetV2 (~3.4 M parameters)
- **Brief description**  
  Mobile-optimized architecture using inverted residuals and linear bottlenecks.

- **Year of ImageNet result**  
  2018

- **Research Paper**  
  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)

- **Novelty**  
  Inverted residual blocks and linear bottlenecks tailored for mobile efficiency.

- **Advantages**  
  Low-latency, small parameter count (~3.4M), high accuracy–speed tradeoff for mobile.

- **Disadvantages**  
  Still requires KPU/GPU optimized deployment; limited capacity compared to very large models.

- **Problems solved**  
  Improved upon tiny model efficiency compared to SqueezeNet, retaining accuracy while being highly mobile-friendly.

---

## 8. SENet (Squeeze-and-Excitation Networks) (various sizes)
- **Brief description**  
  Adds channel-wise attention (Squeeze-and-Excitation blocks) to enhance representational power.

- **Year of ImageNet result**  
  2017 (SENet won ILSVRC classification 1st place) — 2017

- **Research Paper**  
  [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

- **Novelty**  
  Learnable channel-wise recalibration to emphasize informative features.

- **Advantages**  
  Boosts performance of various backbones (e.g. ResNet, Inception) with marginal overhead.

- **Disadvantages**  
  Extra computations and parameters, though modest.

- **Problems solved**  
  Enhanced model expressivity without redesigning entire architectures; improved upon preceding models by focusing on channel interdependencies.

---

## Summary Table
<table>
  <thead>
    <tr>
      <th>Year</th>
      <th>Model</th>
      <th>Params</th>
      <th>Key Improvements</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2012</td>
      <td>AlexNet</td>
      <td>~61 M</td>
      <td>First deep CNN, GPU training, ReLU, dropout</td>
    </tr>
    <tr>
      <td>2014</td>
      <td>GoogLeNet</td>
      <td>~6.8 M</td>
      <td>Inception modules, multi-scale feature extraction</td>
    </tr>
    <tr>
      <td>2014</td>
      <td>VGG16</td>
      <td>~138 M</td>
      <td>Deep uniform small-kernel stack</td>
    </tr>
    <tr>
      <td>2015</td>
      <td>ResNet-50</td>
      <td>~25.6 M</td>
      <td>Residual learning for very deep networks</td>
    </tr>
    <tr>
      <td>2016</td>
      <td>DenseNet-121</td>
      <td>~8 M</td>
      <td>Dense connectivity, feature reuse</td>
    </tr>
    <tr>
      <td>2016</td>
      <td>SqueezeNet</td>
      <td>~1.2 M</td>
      <td>Compact “fire” modules for extreme parameter savings</td>
    </tr>
    <tr>
      <td>2017</td>
      <td>SENet</td>
      <td>varies</td>
      <td>Channel-wise attention via squeeze-and-excitation</td>
    </tr>
    <tr>
      <td>2018</td>
      <td>MobileNetV2</td>
      <td>~3.4 M</td>
      <td>Mobile-friendly inverted residuals and bottlenecks</td>
    </tr>
  </tbody>
</table>


---

Let me know if you'd like help adding performance tables, diagrams, or extending this to more recent models (e.g. EfficientNet, Vision Transformers, etc.).
