This repository contains a TensorFlow implementation of [Kolmogorov-Arnold Network](https://arxiv.org/abs/2404.19756).

<br>

The modules in `layers` are similar to those in `tf.keras.layers`, so you can use these layers as independent modules to assemble the model (in a TensorFlow style):

```python
from tfkan.layers import DenseKAN
# create model using KAN
model = tf.keras.models.Sequential([
    DenseKAN(4),
    DenseKAN(1)
])
model.build(input_shape=(None, 10))
```

When calling `model.summary()` you can see the model structure and its trainable parameters:

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_kan (DenseKAN)        (None, 4)                 484       
                                                                 
 dense_kan_1 (DenseKAN)      (None, 1)                 85        
                                                                 
=================================================================
Total params: 569 (2.22 KB)
Trainable params: 401 (1.57 KB)
Non-trainable params: 168 (672.00 Byte)
_________________________________________________________________
```

When getting started quickly, we can define the optimizer and loss function used for training through `model.compile()` and `model.fit()`, or you can use `tf.GradientTape()` to more finely control the model training behavior and parameter update logic. All model behaviors are the same in `Tensorflow2`, and you can truly use `KAN` as an independent module:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss='mse', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=100, batch_size=64)
```

<br>

The layers currently implemented include:
- `DenseKAN`
    - The basic dense layer, corresponding to `tf.keras.layers.Dense()` (or `nn.Linear()` in torch) in MLP
    - Implement the computational logic described in the KANs paper
    - **grid update method** is available but it will not be automatically used

- `ConvolutionKAN`
    - The basic convolution layer, provide `Conv1DKAN`, `Conv2DKAN` and `Conv3DKAN` for classical 1D, 2D and 3D convolution operations
    - The implementation logic is similar to `Conv` in `Tensorflow`, expanding convolution operations into matrix multiplication, and then replacing MLP dense kernel with `DenseKAN` kernel

<br>

The **grid adaptive update** is an important feature mentioned in KANs paper. In this tensorflow implementation of KANs, **each KAN layer has two method** used to implement this feature:
- `self.update_grid_from_samples(...)`
    - adaptively update the spline grid points using input samples given
    - this can help the spline grid adapt to the numerical range of the input, **avoiding the occurrence of input features outside the support set of the grid** (resulting in outputs that are equal to 0, as spline functions outside the support set are not defined)
- `self.extend_grid_from_samples(...)` 
    - extend the spline grid for given `gird_size`
    - this can help to make the spline activation output more smooth, thereby enhancing the model approximation ability

You can call it in **custom training logic** or use Tensorflow `Callbacks`;

- In custom training logic:
```python
def train_model(...):
    # training logic
    ...

    # call update_grid_from_samples
    for layer in model.layers:
        if hasattr(layer, 'update_grid_from_samples'):
            layer.update_grid_from_samples(x)
        x = layer(x)
```

- or use Tensorflow `Callbacks`:
```python
# define update grid callback
class UpdateGridCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, batch_size: int = 128):
        super(UpdateGridCallback, self).__init__()
        self.training_data = tf.constant(training_data[:batch_size], tf.float32)
        
    def on_epoch_begin(self, epoch, logs=None):
        """
        update grid before new epoch begins
        """
        x_batch = self.training_data
        if epoch > 0:
            for layer in self.model.layers:
                if hasattr(layer, 'update_grid_from_samples'):
                    layer.update_grid_from_samples(x_batch)
                x_batch = layer(x_batch)
```
then add it into `model.fit()`.

## Installation & usage

1. Clone and navigate to the repository folder:
    ```bash
    git clone git@github.com:avrtt/kolmogorov-arnold-network.git
    cd kolmogorov-arnold-network
    ```

2. Run:
    ```bash
    pip install .
    ```

3. Use as packages: 

    ```python
    from kolmogorov-arnold-network import layers
    from kolmogorov-arnold-network.layers import DenseKAN, Conv2DKAN
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.