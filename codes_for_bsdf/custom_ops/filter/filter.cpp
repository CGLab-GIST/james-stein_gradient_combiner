#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cuda_atrous_forward(torch::Tensor input, torch::Tensor param, int numIter);
torch::Tensor cuda_atrous_1D_forward(torch::Tensor input, torch::Tensor param, int numIter);

torch::Tensor cuda_gaussain_forward(torch::Tensor input, int winSize);
torch::Tensor cuda_gaussain_1D_forward(torch::Tensor input, int winSize);



torch::Tensor atrous_filter(
    torch::Tensor input,
    torch::Tensor param,
    int numIter
) {

    CHECK_INPUT(input);
    CHECK_INPUT(param);

    return cuda_atrous_forward(input, param, numIter);
}

torch::Tensor atrous_1D_filter(
    torch::Tensor input,
    torch::Tensor param,
    int numIter
) {

    CHECK_INPUT(input);
    CHECK_INPUT(param);

    return cuda_atrous_1D_forward(input, param, numIter);
}

torch::Tensor gaussian_filter(
    torch::Tensor input,
    int winSize
) {

    CHECK_INPUT(input);


    return cuda_gaussain_forward(input, winSize);
}

torch::Tensor gaussian_1D_filter(
    torch::Tensor input,
    int winSize
) {

    CHECK_INPUT(input);

    return cuda_gaussain_1D_forward(input, winSize);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("atrous_filter", &atrous_filter, "atrous_filter");
    m.def("atrous_1D_filter", &atrous_1D_filter, "atrous_1D_filter");
    m.def("gaussian_filter", &gaussian_filter, "gaussian_filter");
    m.def("gaussian_1D_filter", &gaussian_1D_filter, "gaussian_1D_filter");

}
