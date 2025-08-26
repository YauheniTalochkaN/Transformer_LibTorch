#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

#include <yaml-cpp/yaml.h>

#include <torch/optim.h>
#include <torch/script.h>
#include <torch/nn/functional.h>

#include "CustomTransformer.hh"

int main(int argc, char* argv[]) 
{
    auto model = CustomTransformer(/*input_dim=*/5,
                                   /*d_model=*/128,
                                   /*num_heads=*/8,
                                   /*dim_feedforward=*/512,
                                   /*num_encoder_layers=*/3,
                                   /*num_decoder_layers=*/3,
                                   /*output_dim=*/1,
                                   /*pred_len=*/24);
    model->to(torch::kCUDA);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    int batch_size = 32;
    int src_seq_len = 100;

    torch::Tensor src_train = torch::randn({batch_size, src_seq_len, 5}, options);
    torch::Tensor tgt_train = torch::randn({batch_size, 24, 1}, options);
    torch::Tensor output_train = model->forward(src_train, tgt_train);

    torch::Tensor src_test = torch::randn({batch_size, src_seq_len, 5}, options);
    torch::Tensor output_test = model->forward(src_test);

    std::cout << output_test << "\n";

    return 0;
}