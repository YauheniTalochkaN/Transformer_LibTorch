#pragma once 

#include <iostream>
#include <stdexcept>
#include <vector>

#include <torch/torch.h>

#include "PositionalEncoding.hh"

class CustomTransformerImpl : public torch::nn::Module 
{
public:
    CustomTransformerImpl(int input_dim,
                              int d_model,
                              int num_heads,
                              int dim_feedforward,
                              int num_encoder_layers,
                              int num_decoder_layers,
                              int output_dim,
                              int pred_len = 1,
                              int max_len = 100,
                              float dropout = 0.1);
    virtual ~CustomTransformerImpl() override = default;
    virtual torch::Tensor forward(torch::Tensor src, torch::Tensor tgt = torch::Tensor());

protected:
    int pred_len;
    
    torch::nn::Linear input_projection{nullptr};
    torch::nn::Linear tgt_projection{nullptr};
    PositionalEncoding positional_encoding{nullptr};
    torch::nn::TransformerEncoder transformer_encoder{nullptr};
    torch::nn::TransformerDecoder transformer_decoder{nullptr};
    torch::nn::Linear output_projection{nullptr};

    virtual void init_weights();
    virtual torch::Tensor generate_square_subsequent_mask(int sz);
};

TORCH_MODULE(CustomTransformer);