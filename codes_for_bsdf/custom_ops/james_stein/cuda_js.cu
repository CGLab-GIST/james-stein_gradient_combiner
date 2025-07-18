#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

__forceinline__ __host__ __device__ int iDivUp(int a, int b) { 
      return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}


#define OLS_WGT_EP 0.01f
#define OLS_EP 0.1f
#define JS_EP 1e-20f
#define HALF_SERCH_WINDOW 7
#define RLS_FEAT_DIM 3

float4* g_accOut = NULL;
int g_lenAccOut = 0;


__forceinline__ __host__ __device__ float4 operator+(float b, float4 a) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}

__forceinline__ __host__ __device__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

__forceinline__ __host__ __device__ void operator+=(float4 &a, float4 b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

__forceinline__ __host__ __device__ float4 operator-(float4 a, float b) {
    return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}

__forceinline__ __host__ __device__ void operator-=(float4 &a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    a.w -= b;
}

__forceinline__ __host__ __device__ void operator-=(float4& a, float4 b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}


__forceinline__ __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

__forceinline__ __host__ __device__ float4 operator*(float4 a, float b) {
    return make_float4(a.x * b, a.y * b, a.z * b,  a.w * b);
}

__forceinline__ __host__ __device__ float4 operator*(float b, float4 a) {
    return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

__forceinline__ __host__ __device__ void operator*=(float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}



__forceinline__ __host__ __device__ float4 operator/(float4 a, float4 b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z,  a.w / b.w);
}

__forceinline__ __host__ __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__forceinline__ __host__ __device__ float4 fmaxf(float4 a, float4 b) {
    return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

__forceinline__ __host__ __device__ float4 fminf(float4 a, float4 b) {
    return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}


__device__ void init(float& a) {
	a = 0.f;
}

__device__ void init(float4& a) {
	a.x = a.y = a.z = a.w = 0.f;
}

__forceinline__ __device__ float norm2(const float4& a) {
	return (a.x * a.x) + (a.y * a.y) + (a.z * a.z);
}

__forceinline__ __device__ float avg(const float4& a) {
	return (a.x + a.y + a.z) / 3.f;
}

__forceinline__ __host__ __device__ float4 sqrtf(float4 a) {
    return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w));
}


__global__ void OlsFinalizeKernel(float4* _accGrad, float* _outParamGrad, int height, int width, int param_ch) {
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (cx >= width || cy >= height)
		return;
		
	const int cIdx = cy * width + cx;

	float4 tmpParamGrad = _accGrad[cIdx];
	
	float invWgt = 1.f / tmpParamGrad.w;
	
	if (param_ch == 3){ 
		_outParamGrad[cIdx * 3 + 0] = tmpParamGrad.x * invWgt;
		_outParamGrad[cIdx * 3 + 1] = tmpParamGrad.y * invWgt;
		_outParamGrad[cIdx * 3 + 2] = tmpParamGrad.z * invWgt;
	}
	else {
		_outParamGrad[cIdx] = tmpParamGrad.x * invWgt;
	}
}



__global__ void JamesSteinCombinerCV1DwMomentumKernel_1(const float *_unbiased, const float *_biased, const float *_var, const float *_param,
														float *_rho,
														int height, int width, int winSize)
{
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;

	const int halfWinSize = winSize / 2;
	const int winArea = winSize * winSize;

	float SSE = 0.f;
	float MeanVar = 0.f;
	float MaxVar = 0.f;
	float MinVar = 10000.f;
	int m = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy)
	{
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix)
		{
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float &iUnbiased = _unbiased[idx];
			const float &iBiased = _biased[idx];
			const float &iVar = _var[idx];

			SSE += (iUnbiased - iBiased) * (iUnbiased - iBiased);
			MeanVar += iVar;
			if (iVar > MaxVar)
				MaxVar = iVar;
			else if (iVar < MinVar)
				MinVar = iVar;
			++m;
		}
	}
	MeanVar /= m;

	const int numCandidates = 5;

	float candidates[numCandidates] = {MinVar, (MinVar + MeanVar) * 0.5f, MeanVar, (MeanVar + MaxVar) * 0.5f, MaxVar};

	float Df = (m >= 3) ? ((float)m - 2.f) : 1.f;

	// float optOut = 0.f;
	float rho[5];
	rho[0] = (SSE == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * candidates[0]) / SSE);
	rho[1] = (SSE == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * candidates[1]) / SSE);
	rho[2] = (SSE == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * candidates[2]) / SSE);
	rho[3] = (SSE == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * candidates[3]) / SSE);
	rho[4] = (SSE == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * candidates[4]) / SSE);

	// float combined[5];
	float currMSE[numCandidates] = {0.f, 0.f, 0.f, 0.f, 0.f};

	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy)
	{
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix)
		{
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float &iUnbiased = _unbiased[idx];
			const float &iBiased = _biased[idx];
			const float &iMomentum = _param[idx];
			float iCombined1 = iBiased + rho[0] * (iUnbiased - iBiased);
			float iCombined2 = iBiased + rho[1] * (iUnbiased - iBiased);
			float iCombined3 = iBiased + rho[2] * (iUnbiased - iBiased);
			float iCombined4 = iBiased + rho[3] * (iUnbiased - iBiased);
			float iCombined5 = iBiased + rho[4] * (iUnbiased - iBiased);

			currMSE[0] += (iCombined1 - iMomentum) * (iCombined1 - iMomentum);
			currMSE[1] += (iCombined2 - iMomentum) * (iCombined2 - iMomentum);
			currMSE[2] += (iCombined3 - iMomentum) * (iCombined3 - iMomentum);
			currMSE[3] += (iCombined4 - iMomentum) * (iCombined4 - iMomentum);
			currMSE[4] += (iCombined5 - iMomentum) * (iCombined5 - iMomentum);
		}
	}

	// float cCombined = cBiased + rho * (cUnbiased - cBiased);
	double minMSE = 100000000.0;
	float optRho = 0.f;
	for (int k = 0; k < numCandidates; ++k)
	{
		// double currMSE = (cMomentum - cCombined) * (cMomentum - cCombined);
		if (currMSE[k] < minMSE)
		{
			optRho = rho[k];
			minMSE = currMSE[k];
		}

	} // end k

	_rho[cIdx] = optRho;
}

__global__ void JamesSteinCombinerCV1DwMomentumKernel_2(const float *_unbiased, const float *_biased, const float *_rho,												
														float4 *_accOut,													
														int height, int width, int winSize)
{
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;

	const int halfWinSize = winSize / 2;

	const float &cUnbiased = _unbiased[cIdx];
	const float &cBiased = _biased[cIdx];
	const float cDelta = cUnbiased - cBiased;

	float sum = 0.f;
	float sumw = 0.f;

	// _out[cIdx] = optOut;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy)
	{
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix)
		{
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float iOptRho = _rho[idx];
			float cCombined = cBiased + iOptRho * cDelta;

			sum += cCombined;
			sumw += 1.f;
		}
	}



	_accOut[cIdx].x = sum;
	_accOut[cIdx].w = sumw;
}



__global__ void JamesSteinCombinerCVwMomentumKernel_1(const float *_unbiased, const float *_biased, const float *_var, const float *_param,		
														float *_rho,
														int height, int width, int winSize)
{
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;
	const int halfWinSize = winSize / 2;


	float4 SSE = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 MeanVar = make_float4(0.f, 0.f, 0.f, 0.f);
	float4 MinVar = make_float4(10000.f, 10000.f, 10000.f, 0.f);
	float4 MaxVar = make_float4(0.f, 0.f, 0.f, 0.f);
	int m = 0;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy) {
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix) {
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float4& iUnbiased = make_float4(_unbiased[idx * 3 + 0], _unbiased[idx * 3 + 1], _unbiased[idx * 3 + 2], 0.f);
			const float4& iBiased = make_float4(_biased[idx * 3 + 0], _biased[idx * 3 + 1], _biased[idx * 3 + 2], 0.f);
			const float4& iVar = make_float4(_var[idx * 3 + 0], _var[idx * 3 + 1], _var[idx * 3 + 2], 0.f);
			
			SSE += (iUnbiased - iBiased) * (iUnbiased - iBiased);
			MeanVar += iVar;
			if (avg(iVar) > avg(MaxVar))
				MaxVar = iVar;
			else if (avg(iVar) < avg(MinVar))
				MinVar = iVar;
			++m;
			
	
		}	
	}

	MeanVar.x /= (float)m;
	MeanVar.y /= (float)m;
	MeanVar.z /= (float)m;

	const int numCandidates = 5;
	
	float avgMinVar = avg(MinVar);
	float avgMaxVar = avg(MaxVar);
	float avgMeanVar = avg(MeanVar);

	float candidates[numCandidates] = {avgMinVar, (avgMeanVar + avgMinVar) * 0.5f, avgMeanVar, (avgMeanVar + avgMaxVar) * 0.5f, avgMaxVar};
	// float candidates[numCandidates] = {avgMeanVar};

	double minMSE = 100000000.0;
	float optVar = 0.f;
	float optRho = 0.f;
	float Df = (m > 2.f) ? (m - 2.f) : 1.f;
	for (int k = 0; k < numCandidates; ++k){

		float currVar = candidates[k];
		//float rho = (avg(SSE) == 0.f) ? 1.f : fmaxf(0.f, 1.f - (tau * Df * avg(MeanVar)) / avg(SSE));
		float rho = (avg(SSE) == 0.f) ? 1.f : fmaxf(0.f, 1.f - (Df * currVar) / avg(SSE));
		
		int n = 0;
		float currMSE = 0.f;
		for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy) {
			for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix) {
				int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
				int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
				int idx = y * width + x;
				const float4& iUnbiased = make_float4(_unbiased[idx * 3 + 0], _unbiased[idx * 3 + 1], _unbiased[idx * 3 + 2], 0.f);
				const float4& iBiased = make_float4(_biased[idx * 3 + 0], _biased[idx * 3 + 1], _biased[idx * 3 + 2], 0.f);
				const float4& iMomentum = make_float4(_param[idx * 3 + 0], _param[idx * 3 + 1], _param[idx * 3 + 2], 0.f);
				const float4& iVar = make_float4(_var[idx * 3 + 0], _var[idx * 3 + 1], _var[idx * 3 + 2], 0.f);

				float4 iCombined = iBiased + rho * (iUnbiased - iBiased);
				currMSE += norm2(iCombined - iMomentum) / 3.f;
	
			}	
		}

		if (currMSE < minMSE){
			minMSE = currMSE;
			optRho = rho;
			optVar = currVar;
		}
	
	} //end k
	
	_rho[cIdx * 3 + 0] = optRho;
	_rho[cIdx * 3 + 1] = optRho;
	_rho[cIdx * 3 + 2] = optRho;

	// _rho[cIdx] = optRho;
}

__global__ void JamesSteinCombinerCVwMomentumKernel_2(const float *_unbiased, const float *_biased, const float *_rho,
														float4 *_accOut, int height, int width, int winSize)
{
	const int cx = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	const int cy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
	if (cx >= width || cy >= height)
		return;

	const int cIdx = cy * width + cx;

	const int halfWinSize = winSize / 2;

	const float4 &cUnbiased = make_float4(_unbiased[cIdx * 3 + 0], _unbiased[cIdx * 3 + 1], _unbiased[cIdx * 3 + 2], 0.f);
	const float4 &cBiased = make_float4(_biased[cIdx * 3 + 0], _biased[cIdx * 3 + 1], _biased[cIdx * 3 + 2], 0.f);
	const float4 cDelta = cUnbiased - cBiased;

	// float sum = 0.f;

	float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
	float sumW = 0.f;
	float meanRho = 0.f;
	for (int iy = cy - halfWinSize; iy <= cy + halfWinSize; ++iy)
	{
		for (int ix = cx - halfWinSize; ix <= cx + halfWinSize; ++ix)
		{
			int x = (ix >= width) ? 2 * width - 2 - ix : abs(ix);
			int y = (iy >= height) ? 2 * height - 2 - iy : abs(iy);
			int idx = y * width + x;
			const float& iOptRho = _rho[idx * 3 + 0];

			meanRho += iOptRho;
			float4 cCombined = cBiased + iOptRho * cDelta;

			sum += cCombined;
			sumW += 1.f;
		}
	}


	_accOut[cIdx].x = sum.x;
	_accOut[cIdx].y = sum.y;
	_accOut[cIdx].z = sum.z;
	_accOut[cIdx].w = sumW;
}



std::vector<torch::Tensor> cuda_js_opt_forward(torch::Tensor unbiased, torch::Tensor biased, torch::Tensor var, torch::Tensor param, int winSize)
{

	torch::IntList input_shape = unbiased.sizes();


	torch::Tensor output = torch::zeros_like(unbiased);
	torch::Tensor outRho = torch::zeros_like(unbiased);


	const auto height = input_shape[0];
	const auto width = input_shape[1];
	const auto channel = input_shape[2];

	const int blockDim = 16;
	dim3 threads(blockDim, blockDim);
	dim3 grid(iDivUp(width, blockDim), iDivUp(height, blockDim));

	if (g_lenAccOut < width * height)
	{
		if (g_accOut)
			cudaFree(g_accOut);
		cudaError_t cudaStatus = cudaMalloc((void **)&g_accOut, width * height * sizeof(float4));
		g_lenAccOut = width * height;

		if (cudaStatus != cudaSuccess)
		{
			printf("Err: Malloc failed - Code: %d\n", cudaStatus);
		}
	}


	cudaMemset(g_accOut, 0, width * height * sizeof(float4));

	if (channel == 3)
	{
		JamesSteinCombinerCVwMomentumKernel_1<<<grid, threads>>>(
			unbiased.data<float>(),
			biased.data<float>(),
			var.data<float>(),
			param.data<float>(),
			outRho.data<float>(),
			height, // height
			width,	// width
			winSize);

		JamesSteinCombinerCVwMomentumKernel_2<<<grid, threads>>>(
			unbiased.data<float>(),
			biased.data<float>(),
			outRho.data<float>(),
			g_accOut,
			height, // height
			width,	// width
			winSize);

	}
	else
	{

		JamesSteinCombinerCV1DwMomentumKernel_1<<<grid, threads>>>(
			unbiased.data<float>(),
			biased.data<float>(),
			var.data<float>(),
			param.data<float>(),
			outRho.data<float>(),
			height, // height
			width,	// width
			winSize);

		JamesSteinCombinerCV1DwMomentumKernel_2<<<grid, threads>>>(
			unbiased.data<float>(),
			biased.data<float>(),
			outRho.data<float>(),
			g_accOut,
			height, // height
			width,	// width
			winSize);

	}

	OlsFinalizeKernel<<<grid, threads>>>(
		g_accOut,
		output.data<float>(),
		height,
		width,
		channel);


	return {output, outRho};
}

