#pragma once 

#include <iostream>
#include <stdexcept>

#include <torch/torch.h>

class PositionalEncodingImpl : public torch::nn::Module 
{
public:
    PositionalEncodingImpl(int d_model, int max_len = 100);
    virtual ~PositionalEncodingImpl() override = default;
    virtual torch::Tensor forward(torch::Tensor x);

protected:
    torch::Tensor pe;
};

TORCH_MODULE(PositionalEncoding);