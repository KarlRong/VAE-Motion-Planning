#include <torch/torch.h>
#include <iostream>

using namespace std;
int main()
{
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        cout << "using gpu" << endl;
        device_type = torch::kCUDA;
    }
    else
    {
        cout << "using cpu" << endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    torch::Tensor tensor = torch::rand({2, 3});
    tensor = tensor.to(device);
    std::cout << tensor << std::endl;
}