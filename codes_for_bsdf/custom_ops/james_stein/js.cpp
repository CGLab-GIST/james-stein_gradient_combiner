#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> cuda_js_opt_forward(torch::Tensor unbiased, torch::Tensor biased, torch::Tensor var, torch::Tensor param, int winSize);


std::vector<torch::Tensor> js_opt_forward(
    torch::Tensor unbiased, torch::Tensor biased, torch::Tensor var, torch::Tensor param, int win_size
){
    CHECK_INPUT(unbiased);
    CHECK_INPUT(biased);
    CHECK_INPUT(var);
    CHECK_INPUT(param);

    return cuda_js_opt_forward(unbiased, biased, var, param, win_size);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("js_opt_forward", &js_opt_forward, "js_opt_forward");
}

