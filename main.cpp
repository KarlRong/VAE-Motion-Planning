#include <iostream>
#include <torch/torch.h>

int main(int, char**) {
    torch::Tensor tensor = tensor::eye(3);
    std::cout << tensor << std::endl;
    // std::cout << "Hello, world!\n";
}
