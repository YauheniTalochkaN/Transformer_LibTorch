#include "TimeSeriesTransformer.hh"

TimeSeriesTransformerImpl::TimeSeriesTransformerImpl(int input_dim,
                                                     int d_model,
                                                     int num_heads,
                                                     int dim_feedforward,
                                                     int num_encoder_layers,
                                                     int num_decoder_layers,
                                                     int output_dim,
                                                     int pred_len,
                                                     float dropout) : pred_len(pred_len)
{
    input_projection = register_module("input_projection", torch::nn::Linear(input_dim, d_model));
    
    tgt_projection = register_module("tgt_projection", torch::nn::Linear(output_dim, d_model));
    
    positional_encoding = register_module("positional_encoding", PositionalEncoding(d_model));
    
    torch::nn::TransformerEncoderLayerOptions encoder_layer_options(d_model, num_heads);
    encoder_layer_options.dim_feedforward(dim_feedforward).dropout(dropout);
    
    transformer_encoder = register_module("transformer_encoder",
                                          torch::nn::TransformerEncoder(torch::nn::TransformerEncoderLayer(encoder_layer_options),
                                                                        num_encoder_layers));

    torch::nn::TransformerDecoderLayerOptions decoder_layer_options(d_model, num_heads);
    decoder_layer_options.dim_feedforward(dim_feedforward).dropout(dropout);
    
    transformer_decoder = register_module("transformer_decoder", 
                                          torch::nn::TransformerDecoder(torch::nn::TransformerDecoderLayer(decoder_layer_options),
                                          num_decoder_layers));
    
    output_projection = register_module("output_projection", torch::nn::Linear(d_model, output_dim));
    
    init_weights();
}

void TimeSeriesTransformerImpl::init_weights() 
{
    for (auto& param : parameters()) 
    {
        if (param.dim() > 1) 
        {
            torch::nn::init::xavier_uniform_(param);
        }
    }
}

torch::Tensor TimeSeriesTransformerImpl::generate_square_subsequent_mask(int sz) 
{
    return torch::triu(torch::full({sz, sz}, -std::numeric_limits<float>::infinity()), 1);
}

torch::Tensor TimeSeriesTransformerImpl::forward(torch::Tensor src, torch::Tensor tgt) 
{
    src = input_projection->forward(src);
    src = positional_encoding->forward(src);

    torch::Tensor memory = transformer_encoder->forward(src.transpose(0, 1));
    
    if (!tgt.defined()) 
    {        
        torch::Tensor tgt = torch::zeros({src.size(0), 1, tgt_projection->options.in_features()},
                                         torch::TensorOptions().device(src.device()));
        
        std::vector<torch::Tensor> outputs;
        
        for (int i = 0; i < pred_len; ++i) 
        {            
            torch::Tensor tgt_embed = tgt_projection->forward(tgt);
            tgt_embed = positional_encoding->forward(tgt_embed);
            
            torch::Tensor tgt_mask = generate_square_subsequent_mask(tgt_embed.size(1)).to(src.device());
            
            torch::Tensor output = transformer_decoder->forward(tgt_embed.transpose(0, 1), 
                                                                memory,
                                                                tgt_mask);

            output = output.transpose(0, 1);
            
            torch::Tensor next_step = output_projection->forward(output.index({torch::indexing::Slice(), 
                                                                               torch::indexing::Slice(-1, torch::indexing::None), 
                                                                               torch::indexing::Slice()}));
            
            outputs.push_back(next_step);

            tgt = torch::cat({tgt, next_step}, 1);
        }
        
        return torch::cat(outputs, 1);
    } 
    else 
    {       
        torch::Tensor tgt_embed = tgt_projection->forward(tgt);
        tgt_embed = positional_encoding->forward(tgt_embed);
        
        torch::Tensor tgt_mask = generate_square_subsequent_mask(tgt_embed.size(1)).to(src.device());
        
        torch::Tensor output = transformer_decoder->forward(tgt_embed.transpose(0, 1), 
                                                            memory,
                                                            tgt_mask);

        output = output.transpose(0, 1);

        return output_projection->forward(output);
    }
}