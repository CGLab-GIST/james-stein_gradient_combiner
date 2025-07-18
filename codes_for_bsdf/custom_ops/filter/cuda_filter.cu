#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

__forceinline__ __host__ __device__ int iDivUp(int a, int b) { 
      return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}

#define FLT_EPS 0.0001f

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __host__ __device__ double4 operator*(double4 a, double b) {
    return make_double4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __device__ float Dot(const float4& a, const float4& b) {
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ double4 operator-(double4 a, double4 b) {
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float a, float4 b) {
    return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}


__forceinline__ __host__ __device__ void operator+=(double4 &a, double4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __host__ __device__ void operator/=(float4 a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

__forceinline__ __host__ __device__ void operator/=(double4 a, double b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    a.w /= b;
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}


__forceinline__ __device__ double norm2(double4 a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}


__forceinline__ __device__ float4 logTrans(const float4& a) {
	float4 outCol;
	outCol.x = __logf(a.x + 1.f);
	outCol.y = __logf(a.y + 1.f);
	outCol.z = __logf(a.z + 1.f);
	return outCol;
}



__global__ void GaussianFilter1DKernel(const float* _input, 
							 float* _output, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;

	float stddev = (float)(halfWinSize / 2.f);

    int sy = cy - halfWinSize;
	int sx = cx - halfWinSize;
	int ey = cy + halfWinSize;
	int ex = cx + halfWinSize;
	
	float accData = 0.f;
    float totalWgt = 0.f;
	for (int iy = sy; iy <= ey; iy++) {
		for (int ix = sx; ix <= ex; ix++) {
            if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                continue;
			
			int idx = iy * width + ix;

			const float& iInput = _input[idx];

			float dist2_pos = ((cx - ix) * (cx - ix) + (cy - iy) * (cy - iy)) / (2.f * stddev * stddev);
			float wgt = __expf(-dist2_pos) / (2.f * M_PI * stddev * stddev);

            accData += iInput * wgt;
            totalWgt += wgt;
		}
	}

    float invWgt = 1.f / fmaxf(FLT_EPS, totalWgt);
	
    _output[cIdx] = accData * invWgt;

}

__global__ void GaussianFilterKernel(const float* _input, 
							 float* _output, int height, int width, int winSize) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int nPix = width * height;
	const int halfWinSize = winSize / 2;

	float stddev = (float)(halfWinSize / 2.f);

    int sy = cy - halfWinSize;
	int sx = cx - halfWinSize;
	int ey = cy + halfWinSize;
	int ex = cx + halfWinSize;
	
	float4 accData = make_float4(0.f, 0.f, 0.f, 0.f);
    float totalWgt = 0.f;
	for (int iy = sy; iy <= ey; iy++) {
		for (int ix = sx; ix <= ex; ix++) {
            if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                continue;
			
			int idx = iy * width + ix;

			const float4& iInput = make_float4(_input[idx * 3 + 0], _input[idx * 3 + 1], _input[idx * 3 + 2], 0.f);

			float dist2_pos = ((cx - ix) * (cx - ix) + (cy - iy) * (cy - iy)) / (2.f * stddev * stddev);
			float wgt = __expf(-dist2_pos) / (2.f * M_PI * stddev * stddev);

            accData += iInput * wgt;
            totalWgt += wgt;
		}
	}

    float invWgt = 1.f / fmaxf(FLT_EPS, totalWgt);
	
    _output[cIdx * 3 + 0] = accData.x * invWgt;
	_output[cIdx * 3 + 1] = accData.y * invWgt;
	_output[cIdx * 3 + 2] = accData.z * invWgt;

}


// We use the cross-bilateral filters provieded by Chang et al. 2024
// Wesley Chang, Xuanda Yang, Yash Belhe, Ravi Ramamoorthi, and Tzu-Mao Li. 2024. Spatiotemporal Bilateral Gradient Filtering for Inverse Rendering. 
// In SIGGRAPH Asia 2024 Conference Papers (SA '24). Association for Computing Machinery, New York, NY, USA, Article 70, 1-11. 


__global__ void filter(
    const float* input_grad,
    const float* input_primal,
    float* output,
    int height,
    int width,
	int iter
    ){
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;


    double eps = 1e-10;
	double sigma_d = 0.1;
    double kernel[] = {3.f / 8, 1.f / 4, 1.f / 16};

	int stride = pow(2, iter);

	double4 param = make_double4((double)input_primal[cIdx * 3 + 0], (double)input_primal[cIdx * 3 + 1], (double)input_primal[cIdx* 3 + 2], 0.0);
	double w_sum = 0.0f;
	double4 filtered_grad = make_double4(0.0, 0.0, 0.0, 0.0);
    int r = 2;
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r ; ++x) {

			int offset[2] = {x * stride, y * stride};
			int iy = cy + offset[1];
			int ix = cx + offset[0];
	

            if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                continue;

			int idx = iy * width + ix;
			double4 grad_neighbor = make_double4((double)input_grad[idx * 3 + 0], (double)input_grad[idx * 3 + 1], (double)input_grad[idx * 3 + 2], 0.0);
			double4 param_neighbor = make_double4((double)input_primal[idx * 3 + 0], (double)input_primal[idx * 3 + 1], (double)input_primal[idx * 3 + 2], 0.0);
			double dist2_param = sqrt(norm2(param - param_neighbor));
            double w_l = exp(-dist2_param / (sigma_d + eps));
            double h = kernel[abs(x)] * kernel[abs(y)];
            double w = h * w_l;

            filtered_grad += grad_neighbor * w;
            w_sum += w;
        }
    }

    filtered_grad /= w_sum;

    output[cIdx * 3 + 0] = filtered_grad.x;
	output[cIdx * 3 + 1] = filtered_grad.y;
	output[cIdx * 3 + 2] = filtered_grad.z;
}


__global__ void filter1D(
    const float* input_grad,
    const float* input_primal,
    float* output,
    int height,
    int width,
	int iter
    ){
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;
	const int cIdx = cy * width + cx;

    
    double eps = 1e-30;
	double sigma_d = 0.1;
    double kernel[] = {3.0 / 8, 1.0 / 4, 1.0 / 16};

	
	int stride = pow(2, iter);
	double param = (double)input_primal[cIdx];

	double w_sum = 0.0f;
	double filtered_grad = 0.f;
    int r = 2;
    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r ; ++x) {
			int offset[2] = {x * stride, y * stride};
			int iy = cy + offset[1];
			int ix = cx + offset[0];
            if (ix < 0 || ix >= width || iy < 0 || iy >= height)
                continue;

			int idx = iy * width + ix;
			float grad_neighbor = input_grad[idx];
			double param_neighbor = (double)input_primal[idx];

			double diff_param = (param - param_neighbor);

            double w_l = exp(-sqrt(diff_param * diff_param)/ (sigma_d + eps));
            
			double h = kernel[abs(x)] * kernel[abs(y)];
            double w = w_l * h;

            filtered_grad += w * (double)grad_neighbor;
            w_sum += w;
        }
    }

    filtered_grad /= w_sum;

    output[cIdx] = filtered_grad;
}



torch::Tensor cuda_atrous_forward(torch::Tensor input, torch::Tensor param, int numIter)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	torch::Tensor output = torch::zeros_like(input);
	torch::Tensor tmpInput = input;

	for (int iter = 0; iter < numIter; ++iter){

		output = torch::zeros_like(input);
		filter <<<grid, threads>>> (
        tmpInput.data<float>(),
        param.data<float>(),
        output.data<float>(),
        height, 
        width,
		iter
        );

		tmpInput = output;

	}


    return output;
}

torch::Tensor cuda_atrous_1D_forward(torch::Tensor input, torch::Tensor param, int numIter)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	torch::Tensor output = torch::zeros_like(input);
	torch::Tensor tmpInput = input;

	for (int iter = 0; iter < numIter; ++iter){

		output = torch::zeros_like(input);
		filter1D <<<grid, threads>>> (
        tmpInput.data<float>(),
        param.data<float>(),
        output.data<float>(),
        height,
        width,
		iter
        );

		tmpInput = output;

	}


    return output;
}


torch::Tensor cuda_gaussain_forward(torch::Tensor input, int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	torch::Tensor output = torch::zeros_like(input);


	GaussianFilterKernel <<<grid, threads>>> (
        input.data<float>(),
        output.data<float>(),
        height,
        width,
		winSize
        );

    return output;
}


torch::Tensor cuda_gaussain_1D_forward(torch::Tensor input, int winSize)
{
    // Get tensor shapes
    torch::IntList input_shape = input.sizes();

	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

	// torch::Tensor 
    const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	torch::Tensor output = torch::zeros_like(input);

	GaussianFilter1DKernel <<<grid, threads>>> (
        input.data<float>(),
        output.data<float>(),
        height,
        width,
		winSize
        );

    return output;
}


