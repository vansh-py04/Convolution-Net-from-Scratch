🧠 ThreeLayerConvNet
A from-scratch implementation of a three-layer convolutional neural network (CNN) in NumPy with custom forward and backward passes for convolution, pooling, activation, affine layers, and softmax loss.

🔧 Architecture

    Input → Conv → ReLU → MaxPool(2x2) → Affine → ReLU → Affine → Softmax
This minimal CNN supports:

Custom weight initialization

Naive forward/backward implementations (no deep learning libraries used)

Layer composability through modular design

Debugging-friendly shape printing for each major block

🙋‍♂️ Credits
Built with by [Yuganter Pratap](https://www.linkedin.com/in/yuganter-pratap-a3a719254/) — inspired by CS231n and DIY deep learning educational projects.
