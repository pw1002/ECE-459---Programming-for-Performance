extern "C" __global__ void convolution_layer(double input[100][100],double filter[10][5][5],double output[10][20][20]) {
    double res = 0;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int filter_num = blockIdx.x;

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            res += filter[filter_num][i][j] * input[x*5 + i][y*5 + j];
        }
    }
    output[filter_num][x][y] = res;
}

extern "C" __global__ void relu_layer(double input[10][20][20],double output[10][20][20]) {
    int x = threadIdx.x;
    int y = threadIdx.y;
    int filter_num = blockIdx.x;

    output[filter_num][x][y] = (input[filter_num][x][y] < 0) ? 0.0 : input[filter_num][x][y];
}

extern "C" __global__ void output_layer(double input[4000],double weight[10][4000],double output[10][16]) {
    double res = 0;
    int thread = threadIdx.x;
    int weight_num = blockIdx.x;

    for (int i = 0; i < 250; i++) {
        int inp = i + thread*250;
        res += input[inp] * weight[weight_num][inp];
    }

    output[weight_num][thread] = res;
}
