// 示例CUDA Kernel - 向量加法
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 示例CUDA Kernel - 矩阵乘法
__global__ void matrixMul(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

// 示例CUDA Kernel - ReLU激活
__global__ void relu(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (x[idx] < 0) {
            x[idx] = 0;
        }
    }
}

// 示例CUDA Kernel - Softmax
__global__ void softmax(float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float max_val = -INFINITY;
        for (int i = 0; i < n; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += expf(x[i] - max_val);
        }
        
        y[idx] = expf(x[idx] - max_val) / sum;
    }
}
