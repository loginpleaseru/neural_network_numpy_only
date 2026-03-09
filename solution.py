from interface import *
import scipy 


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            momentum_grad = self.lr * parameter_grad + self.momentum * updater.inertia 
            updater.inertia = momentum_grad
            return parameter - momentum_grad
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return inputs.clip(0,None)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        mask = self.forward_inputs >= 0
        return (grad_outputs * mask)
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        inputs_offset = inputs.copy()
        offset = inputs_offset.max(axis=1, keepdims = True)
        inputs_offset -= offset
        exp = np.exp(inputs_offset)
        norm_exp = exp.sum(axis=1, keepdims=True)
        output = exp/norm_exp
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        softmax = self.forward_impl(self.forward_inputs)
        return grad_outputs * softmax - softmax * (grad_outputs * softmax).sum(axis=1).reshape(-1,1)
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return  inputs @ self.weights.T + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.weights_grad = (grad_outputs.T @ self.forward_inputs)
        self.biases_grad  = grad_outputs.sum(axis=0).flatten() 
        return grad_outputs @ self.weights
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        y_pred = y_pred.clip(np.finfo(float).eps, None)
        return -np.mean( (y_gt * np.log(y_pred)).sum(axis=1),keepdims=True)
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        
        y_pred = y_pred.clip(np.finfo(float).eps, None)
        return (-(y_gt/y_pred))/y_gt.shape[0]
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    optimizer = SGD(lr=1e-3)
    loss = CategoricalCrossentropy()
    model =  Model(loss, optimizer)
    model.add(Dense(input_shape=(784,), units=784))
    model.add(ReLU())
    model.add(Dense(units=784))
    model.add(ReLU())
    model.add(Dense(units=500))
    model.add(ReLU())
    model.add(Dense(units=100))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=256, epochs=10, shuffle=False, verbose=True, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    kernels = np.rot90(kernels, 2, axes=(2,3))
    padded_inputs = np.pad(inputs, pad_width=((0,0),(0,0), (padding,padding),(padding,padding))).copy()
    batch_size, d, ih, iw = inputs.shape
    count_kernels, d, kh, kw = kernels.shape
    oh = inputs.shape[2] - kernels.shape[2] + 2*padding + 1
    ow = inputs.shape[3] - kernels.shape[3] + 2*padding + 1
    patch_kernels = kernels[np.newaxis, ...]
    

    output_img = np.zeros(shape=(batch_size,count_kernels,oh,ow))
    for i in range(oh):
        for j in range(ow):
            patch_img = padded_inputs[:, : , i:i+kh, j:j+kw]
            patch_img = patch_img[:, np.newaxis, ...]
            output_img[:,:,i,j] = (patch_img * patch_kernels).sum(axis=(2,3,4))
    return output_img



# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
    
        # your code here \/
        self.inputs = inputs
        return convolve(inputs, self.kernels, self.kernel_size//2) + self.biases[np.newaxis, :, np.newaxis,np.newaxis]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        self.biases_grad = grad_outputs.sum(axis=(0,2,3))
        inputs_for_grad = self.inputs.transpose(1,0,2,3)
        kernel_for_grad = grad_outputs.transpose(1,0,2,3)
        kernel_for_grad = np.rot90(kernel_for_grad, 2, axes=(2,3))
        pre_kernel_grad = convolve(inputs_for_grad,kernel_for_grad, padding=self.kernel_size//2)
        self.kernels_grad = np.rot90(pre_kernel_grad, 2, axes=(2,3)).transpose(1,0,2,3)
        inputs_grad = convolve(grad_outputs, np.rot90(self.kernels.transpose(1,0,2,3), 2, axes=(2,3)), padding = self.kernel_size//2)

        return inputs_grad
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        n,d, ih, iw = inputs.shape
        self.ih = ih
        self.iw = iw
        self.inputs = inputs
        kernelled_input = inputs.reshape(n,d,ih//self.pool_size, self.pool_size, iw//self.pool_size, self.pool_size)


        if self.pool_mode == 'max':
            output = kernelled_input.max(axis=(3,5))
            self.output = output
        else:
            output = np.mean(kernelled_input,axis=(3,5))
            self.output = output
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        n, d, oh, ow = grad_outputs.shape
        output_of_input_dim = np.repeat(np.repeat(self.output,self.pool_size,axis=2),self.pool_size,axis=3)
        grad_of_input_dim = np.repeat(np.repeat(grad_outputs,self.pool_size,axis=2),self.pool_size,axis=3)


        if self.pool_mode == 'max':
            mask = (output_of_input_dim == self.inputs) ##Здесь боремся с тем, чтобы True выдавался на один максимальный элемент из ядра. Потому что если в окне два максимальных значения, то значение прилетит им обоим, а это плохо
            kernelled_mask = mask.reshape(n,d,self.ih//self.pool_size,self.pool_size, self.iw//self.pool_size,self.pool_size)
            kernelled_mask = kernelled_mask.transpose(0,1,2,4,3,5)
            kernelled_mask = kernelled_mask.reshape(n,d,self.ih//self.pool_size, self.iw//self.pool_size, self.pool_size * self.pool_size)
            argmax_mask = kernelled_mask.argmax(axis=4)
            zeros_mask = np.zeros((n,d,self.ih//self.pool_size, self.iw//self.pool_size, self.pool_size * self.pool_size))
            np.put_along_axis(zeros_mask, argmax_mask[...,None],1,axis=-1)
            zeros_mask = zeros_mask.reshape(n,d,self.ih//self.pool_size, self.iw//self.pool_size, self.pool_size , self.pool_size)
            zeros_mask = zeros_mask.transpose(0,1,2,4,3,5)
            zeros_mask = zeros_mask.reshape(n, d, self.ih, self.iw)


            dl_di = (zeros_mask * grad_of_input_dim) 
            
        elif self.pool_mode == 'avg':
            dl_di = (grad_of_input_dim / (self.pool_size * self.pool_size)).astype(np.float64)
        return dl_di
        


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        if self.is_training:
            self.mean_value = np.mean(inputs, axis=(0,2,3), keepdims=True)
            self.var_value = np.var(inputs, axis=(0,2,3), keepdims=True)

            self.forward_centered_inputs = inputs - self.mean_value
            self.forward_inverse_std = 1/np.sqrt(eps + self.var_value)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum)* self.mean_value.ravel()
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * self.var_value.ravel()

        else:
            self.forward_normalized_inputs = (inputs - self.running_mean[None,:,None,None])/np.sqrt(eps + self.running_var[None,:,None,None])
        return self.gamma[None,:,None,None] * self.forward_normalized_inputs + self.beta[None,:,None,None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        n,d,h,w = grad_outputs.shape
        self.beta_grad = grad_outputs.sum(axis = (0,2,3))
        self.gamma_grad = (grad_outputs * self.forward_normalized_inputs).sum(axis = (0,2,3))
        dLoss_dXnorm = grad_outputs * self.gamma[None, :, None, None]
        N = n*h*w
        dL_xi_1 = dLoss_dXnorm * self.forward_inverse_std
        dL_xi_2 = dLoss_dXnorm.sum(axis=(0,2,3),keepdims=True) * -(self.forward_inverse_std/N)
        dL_xi_3 = -(self.forward_centered_inputs/N) * (self.forward_inverse_std)**3 * (dLoss_dXnorm * self.forward_centered_inputs).sum(axis=(0,2,3),keepdims=True)
        dL_dinput = dL_xi_1 + dL_xi_2 + dL_xi_3
        return dL_dinput
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        self.n,self.d,self.h,self.w = inputs.shape
        to_return = (inputs.reshape(self.n,self.d*self.h*self.w)).astype(np.float64)
        # your code here \/
        return to_return
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        dL_dinputs = grad_outputs.reshape(self.n,self.d,self.h,self.w).astype(np.float64)
        # your code here \/
        return dL_dinputs
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        
        # your code here \/
        if self.is_training:
            random_numbers = np.random.uniform(0,1, inputs.shape)
            random_numbers = random_numbers > self.p
            self.forward_mask = random_numbers
            output = random_numbers * inputs
        else:
            output = (1-self.p) * inputs
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return grad_outputs * self.forward_mask
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    optimizer = SGDMomentum(lr=1e-2, momentum=0.9)
    loss = CategoricalCrossentropy()
    model = Model(loss, optimizer=optimizer)

    model.add(Conv2D(output_channels=16, input_shape=(3,32,32)))
    model.add(ReLU())
    model.add(Pooling2D()) 
    model.add(BatchNorm())

    model.add(Conv2D(output_channels=24)) 
    model.add(ReLU())    
    model.add(Pooling2D())
    model.add(BatchNorm())

    model.add(Conv2D(output_channels=32)) 
    model.add(ReLU())    
    model.add(Pooling2D())
    model.add(BatchNorm())
    model.add(Flatten())

    model.add(Dense(units=256))
    model.add(ReLU())
    model.add(Dropout(p=0.2))

    model.add(Dense(units=128))
    model.add(ReLU())
    model.add(Dropout(p=0.2))
    
    model.add(Dense(units=10))
    model.add(Softmax())


    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=32, epochs=6, shuffle=False, verbose=True, x_valid=x_valid, y_valid=y_valid)

    
    return model


