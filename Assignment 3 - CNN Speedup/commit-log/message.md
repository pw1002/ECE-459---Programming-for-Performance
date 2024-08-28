Implemented GPU usage when calculating CNN

The implemented code was done in tehe cuda.rs and kernel.cu files. The idea here is to initialize cuda and start the kernel in the cuda.rs file and the actualy computations of the different layers are done in the kernel.cu file.
The initialization of cuda is simple and is pretty much from lecture 22. The compute function initializes DeviceBoxes to allow communication to the GPU and also defines the grid and block sizes of the different layers. The kernel.cu file simply take these inputs and does the computations of the convolutions, ReLU and output layers.

The convolution layer is in charge of perofrming dot product between the input and the filter matricies and to parallelize this, we split the computation into grids. The grid size has dimensions 10x1 where each grid corresponds to a specific filter (one for each neuron in the convolution layer). And each block within the grid processes a 20x20 area of the input. Therefore, we can use blockIdx.x to get each filter and threadIdx.x and threadIdx.y to get the elements within each block.

The ReLU layer is quite simple. The dimensions for everything are pretty much the same as the concolution layer since it just changes negative values to 0, but doesn't change any of the dimensions. Here I simply checked whether the output of the convolution layer outputted a number less than 0. And if it is, then we set the output of the corresponding index of the output matrix to 0, otherwise we just set it to the input value.

The output layer is also similar. Here, we also set the grid size to 10 which corresponds to each layer in the weight vector. I also set the number of threads to be 16 here (I also could have used 32 or other numbers here but found 16 to have consistent shorter runtimes). The number of times we loop is simply 4000/16 which is 250 to make sure we loop through all the entries. 

I know this code is correct because when running the compare.py file after running generate.py and then running both the CPU and GPU versions, it doesn't give any errors or show any differences.

The performance of the GPU implementation is on average fastter than the CPU version. Running this 3 times for both the CPU and GPU version I got on average 243925 microseconds for the CPU implenetation and 176960 microseconds for the GPU implementation.