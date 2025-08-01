//std
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "entropy_bottleneck.h"
#include "onnx_model_wrapper.h"
#include "rans_interface.hpp"

#include <filesystem>
namespace fs = std::filesystem;

EntropyBottleNeck::EntropyBottleNeck(const std::string& npz_path){
    // 不存在
    if (!fs::exists(npz_path)) {
        throw std::runtime_error("npz_path " + npz_path + " not found");
    }
    
    auto data = cnpy::npz_load(npz_path);
    cnpy::NpyArray npy_quantized_cdf = data["_quantized_cdf"];
    int C = npy_quantized_cdf.shape[0];
    int bins = npy_quantized_cdf.shape[1];

    std::vector<int> v =  npy_quantized_cdf.as_vec<int>();

    for (int i = 0; i < C; ++i) {
        int* ptr = v.data() + i * bins;
        quantized_cdf_.push_back(std::vector<int>(ptr, ptr + bins));
    }

    cnpy::NpyArray npy_cdf_length = data["_cdf_length"];
    cnpy::NpyArray npy_offset = data["_offset"];
    cnpy::NpyArray npy_quantiles = data["quantiles"];

    cdf_length_ = npy_cdf_length.as_vec<int>();
    offset_ = npy_offset.as_vec<int>();
    quantiles_ = npy_quantiles.as_vec<float>();

    std::cout << "quantized_cdf.shape: " << quantized_cdf_.size() << std::endl;
    std::cout << "cdf_length.shape: " << cdf_length_.size() << std::endl;
    std::cout << "offset.shape: " << offset_.size() << std::endl;
    std::cout << "quantiles.shape: " << quantiles_.size() << std::endl;
}


std::vector<std::string> EntropyBottleNeck::compress(const xt::xarray<float>& input) {
    // dummy input_shape
    std::cout << "input.shape: " << xt::adapt(input.shape()) << std::endl;

    int N = input.shape()[0];
    int C = input.shape()[1];
    int H = input.shape()[2];
    int W = input.shape()[3];


    // adapt vector
    std::vector<size_t> q_shape{static_cast<size_t>(C), 1, 3};
    xt::xarray<float> quantiles_xarray = xt::adapt(quantiles_, q_shape);
    std::cout << "quantiles_xarray.shape: " << xt::adapt(quantiles_xarray.shape()) << std::endl;
    std::cout << "quantiles_xarray.mean: " << xt::mean(quantiles_xarray) << std::endl;
    std::cout << " ---------------------------- " << std::endl;

    xt::xarray<int> index_xarray = xt::zeros<int>({N, C, H, W});
    xt::xarray<float> medians_xarray = xt::zeros<float>({N, C, 1, 1});
    xt::xarray<int> symbol_xarray = xt::zeros<int>({N, C, H, W});


    for(int in = 0; in < N; in++) {
        for(int ic = 0; ic < C; ic++) {
            for(int ih = 0; ih < H; ih++) {
                for(int iw = 0; iw < W; iw++) {
                    index_xarray.at(in, ic, ih, iw) = ic;
                    float q_value = quantiles_xarray.at(ic, 0, 1);
                    medians_xarray.at(in, ic, 0, 0) = q_value;

                    float input_value = input.at(in, ic, ih, iw);
                    symbol_xarray.at(in, ic, ih, iw) = static_cast<int>(std::round(input_value - q_value));
                }
            }
        }
    }

    std::cout << "index.shape: " << xt::adapt(index_xarray.shape()) << std::endl;
    std::cout << "medians.shape: " << xt::adapt(medians_xarray.shape()) << std::endl;
    std::cout << "symbol.shape: " << xt::adapt(symbol_xarray.shape()) << std::endl;
    
    // encode
    std::vector<std::string> strings_list;
    for (int ni = 0; ni < N; ni++) {
        xt::xarray<int> symbol_ni = xt::view(symbol_xarray, xt::range(ni, ni+1), xt::all(), xt::all(), xt::all());
        xt::xarray<int> index_ni = xt::view(index_xarray, xt::range(ni, ni+1), xt::all(), xt::all(), xt::all());

        std::cout << "symbol_ni.shape: " << xt::adapt(symbol_ni.shape()) << std::endl;
        std::cout << "index_ni.shape: " << xt::adapt(index_ni.shape()) << std::endl;

        std::vector<int> symbol_vec(symbol_ni.begin(), symbol_ni.end());
        std::vector<int> index_vec(index_ni.begin(), index_ni.end());

        std::string strings = rans_enc.encode_with_indexes(symbol_vec, 
                                                        index_vec, 
                                                        quantized_cdf_, 
                                                        cdf_length_, 
                                                        offset_);

        std::cout << "strings.size: " << strings.size() << std::endl;
        strings_list.push_back(strings);
    }

    return strings_list;

}

xt::xarray<float> EntropyBottleNeck::decompress(const std::vector<std::string>& strings_list, const std::vector<int>& input_shape) {
    // dummy input_shape
    std::cout << "strings_list.size: " << strings_list.size() << std::endl;
    std::cout << "strings_list[0].size: " << strings_list[0].size() << std::endl;

    int latent_rows = input_shape[0];
    int latent_cols = input_shape[1];
    int C = quantized_cdf_.size();
    int N = strings_list.size();
    int H = latent_rows;
    int W = latent_cols;

    std::vector<int> output_size{1, C, latent_rows, latent_cols};

    std::vector<size_t> q_shape{static_cast<size_t>(C), 1, 3};
    xt::xarray<float> quantiles_xarray = xt::adapt(quantiles_, q_shape);

    xt::xarray<int> index_xarray = xt::zeros<int>({N, C, H, W});
    xt::xarray<float> medians_xarray = xt::zeros<float>({N, C, 1, 1});

    for(int in = 0; in < N; in++) {
        for(int ic = 0; ic < C; ic++) {
            for(int ih = 0; ih < H; ih++) {
                for(int iw = 0; iw < W; iw++) {
                    index_xarray.at(in, ic, ih, iw) = ic;
                    float q_value = quantiles_xarray.at(ic, 0, 1);
                    medians_xarray.at(in, ic, 0, 0) = q_value;
                }
            }
        }
    }


    // decode
    xt::xarray<float> output_xarray = xt::zeros<float>({N, C, H, W});
    for (int ni=0; ni<N; ni++) {
        std::string compressed_string = strings_list[ni];
        xt::xarray<int> index_ni = xt::view(index_xarray, ni, xt::all(), xt::all(), xt::all());
        std::vector<int> index_vec(index_ni.begin(), index_ni.end());
        
        std::vector<int32_t> values = rans_dec.decode_with_indexes(compressed_string, 
                                                                    index_vec, 
                                                                    quantized_cdf_, 
                                                                    cdf_length_, 
                                                                    offset_);
        std::vector<size_t> values_shape{static_cast<size_t>(C), static_cast<size_t>(H), static_cast<size_t>(W)};
        xt::xarray<int> values_xarray = xt::adapt(values, values_shape);

        auto t = xt::view(output_xarray, ni, xt::all(), xt::all(), xt::all());
        t = xt::cast<float>(values_xarray);
    }

    output_xarray = output_xarray + medians_xarray;

    return output_xarray;
}




EntropyBottleNeck::~EntropyBottleNeck() {
}


// int main() {
//     EntropyBottleNeck entropy_bottleneck("bmshj2018-factorized-entropy_bottleneck.npz");
//     xt::xarray<float> input = xt::load_npy<float>("g_a_output.npy");
//     std::vector<std::string> strings = entropy_bottleneck.compress(input);
//     std::cout << "strings.size: " << strings[0].size() << std::endl;
//     return 0;
// }
