#include "PositionalEncoding.hh"

PositionalEncodingImpl::PositionalEncodingImpl(int d_model, int max_len) 
{
    pe = torch::zeros({max_len, d_model});
    
    torch::Tensor position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
    torch::Tensor div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * -(std::log(10000.0) / d_model));
    
    pe.index_put_({torch::indexing::Slice(), 
                   torch::indexing::Slice(0, torch::indexing::None, 2)},
                   torch::sin(position * div_term));
                   
    pe.index_put_({torch::indexing::Slice(), 
                   torch::indexing::Slice(1, torch::indexing::None, 2)},
                   torch::cos(position * div_term));
    
    pe = pe.unsqueeze(0);

    register_buffer("pe", pe);
}

torch::Tensor PositionalEncodingImpl::forward(torch::Tensor x) 
{
    return x + pe.index({torch::indexing::Slice(),
                         torch::indexing::Slice(0, x.size(1)),
                         torch::indexing::Slice()});
}