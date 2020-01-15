#include <torch/torch.h>
#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

using namespace std;

int main(int argc, const char *argv[])
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

    if (argc != 2)
    {
        std::cerr << "usage: narrowpassage <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::cout << "load module " <<  argv[1] <<" ok\n";

    torch::Tensor sample = torch::rand({256, 6}).to(device);
    torch::Tensor startend = torch::rand({256, 12}).to(device);
    torch::Tensor occ = torch::rand({256, 1, 100, 100}).to(device);
    vector<torch::jit::IValue> inputs{sample, startend, occ};
    auto out = module.forward(inputs).toTuple();
    auto output = out->elements();
    auto res = output.at(0).toTensor();
    cout << res.sizes() << endl;
    return 0;
}