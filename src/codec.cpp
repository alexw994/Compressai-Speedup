
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "save_utils.h"
#include "entropy_bottleneck.h"
#include "codec.h"
#include "onnx_model_wrapper.h"

xt::xarray<float> ptr2xarray(const std::shared_ptr<uint8_t>& rgb_data, 
                            uint32_t original_width, uint32_t original_height) {
    size_t total_size = static_cast<size_t>(original_height * original_width * 3);

    // RGB
    auto img_hwc = xt::adapt(rgb_data.get(), 
                    total_size, 
                    xt::no_ownership(), 
                    std::vector<size_t>{original_height, original_width, 3});
    auto img_chw = xt::transpose(img_hwc, {2, 0, 1});
    auto img_1chw = xt::expand_dims(img_chw, 0);
    xt::xarray<float> input_data_4d = xt::eval(img_1chw / 255.0);
    return input_data_4d;
}


auto pad4d(const xt::xarray<float>& x, uint32_t p=64) {
    uint32_t h = xt::adapt(x.shape())[2];
    uint32_t w = xt::adapt(x.shape())[3];
    uint32_t pad_h = (h + p - 1) / p * p;
    uint32_t pad_w = (w + p - 1) / p * p;
    const std::vector<std::vector<uint32_t>> pad_vector = {{0, 0}, {0, 0}, {(pad_h - h) / 2, (pad_h - h) / 2}, {(pad_w - w) / 2, (pad_w - w) / 2}};
    // N C H W
    std::cout << "----------pad_h: " << pad_h << std::endl;
    std::cout << "----------pad_w: " << pad_w << std::endl;
    auto x_pad = xt::pad(x, pad_vector);
    std::cout << "----------x_pad.shape: " << xt::adapt(x_pad.shape()) << std::endl;
    return x_pad;
}

auto crop4d(const xt::xarray<float>& x, uint32_t original_height, uint32_t original_width) {
    uint32_t h = xt::adapt(x.shape())[2];
    uint32_t w = xt::adapt(x.shape())[3];
    uint32_t crop_h = std::min(h, original_height);
    uint32_t crop_w = std::min(w, original_width);
    // N C H W
    auto x_crop = xt::view(x, xt::all(), xt::all(), 
                  xt::range((h - crop_h) / 2, (h - crop_h) / 2 + original_height), 
                  xt::range((w - crop_w) / 2, (w - crop_w) / 2 + original_width));
    return x_crop;
}

void encode_buffer(Params& params, const std::string& model_dir) {
    std::string model_name = params.model_name;
    std::string metric_name = params.metric_name;
    char quality = params.quality;
    uint32_t original_width = params.original_width; // 原始图像宽度
    uint32_t original_height = params.original_height; // 原始图像高度

    if (params.rgb_data == nullptr) {
        throw std::runtime_error("rgb_data is nullptr");
    }

    if (model_name != "bmshj2018-factorized" && model_name != "bmshj2018-factorized_relu") {
        std::cout << "----------model_name: " << model_name << std::endl;
        throw std::runtime_error("model is not supported");
    }
  
    uint32_t Scale = 16;
    
    xt::xarray<float> input_data_4d = ptr2xarray(params.rgb_data, original_width, original_height);
    std::cout << "----------input_data_4d.shape: " << xt::adapt(input_data_4d.shape()) << std::endl;

    xt::xarray<float> input_data_4d_pad = xt::eval(pad4d(input_data_4d, 64));

    std::cout << "----------pad input_data_4d.shape: " << xt::adapt(input_data_4d_pad.shape()) << std::endl;
    uint32_t after_pad_height = input_data_4d_pad.shape()[2];
    uint32_t after_pad_width = input_data_4d_pad.shape()[3];

    //load g_a
    OnnxModelInferenceWrapper g_a(model_dir + "/" + model_name + "-" + metric_name + "-q" + quality + "-g_a.onnx", false);
    uint32_t C = static_cast<uint32_t>(g_a.outputDims_[1]);

    EntropyBottleNeck entropy_bottleneck_wrapper(model_dir + "/" + model_name + "-" + metric_name + "-q" + quality + "-entropy_bottleneck.npz");

    // encode
    // infer g_a
    std::vector<int64_t> input_size = {1, 3, (int)after_pad_height, (int)after_pad_width};
    int output_rows = after_pad_height / Scale;
    int output_cols = after_pad_width / Scale;
    std::vector<int64_t> output_size = {1, C, output_rows, output_cols};

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<float>> outputs = g_a.run(input_data_4d_pad, input_size, output_size);

    std::vector<float> output_data_g_a = outputs[0];
    // infer entropy_bottleneck.compress(y)
    std::vector<size_t> output_shape{1, C, static_cast<size_t>(output_rows), static_cast<size_t>(output_cols)};
    std::size_t total_output_size = static_cast<size_t>(1 * C * output_rows * output_cols);
    auto output_data_g_a_xarray = xt::eval(xt::adapt(output_data_g_a.data(), 
                                                        total_output_size, 
                                                        xt::no_ownership(), output_shape));

    std::vector<std::string> compressed_strings = entropy_bottleneck_wrapper.compress(output_data_g_a_xarray);

    params.compressed_string = compressed_strings[0];
    params.output_rows = output_rows;
    params.output_cols = output_cols;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "encode time taken: " << duration.count() << " milliseconds" << std::endl;
}


void encode_file(const std::string& input_file, const std::string& output_file, Params& params, const std::string& model_dir) {
    // bgr
    cv::Mat input_image = cv::imread(input_file);
    // rgb
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);

    std::shared_ptr<uint8_t> buffer(
        reinterpret_cast<uint8_t*>(input_image.data),
        [input_image](uint8_t*) mutable {
            // capture input_image by copy to hold its memory
            // do nothing on delete, because input_image manages memory
        }
    );

    params.rgb_data = buffer;
    params.original_height = input_image.rows;
    params.original_width = input_image.cols;

    encode_buffer(params, model_dir);

    std::string compressed_string = params.compressed_string;

    char code;
    build_code(metric_ids[params.metric_name], params.quality, code);
    char original_bitdepth = 8;
    uint32_t original_height = params.original_height;
    uint32_t original_width = params.original_width;
    uint32_t output_rows = params.output_rows;
    uint32_t output_cols = params.output_cols;
    uint32_t n_strings = 1;
    std::vector<uint32_t> length_strings = {static_cast<uint32_t>(compressed_string.size())};
    std::vector<std::string> strings = {compressed_string};

    fileInfo finfo = {
        output_file,
        model_ids[params.model_name],
        code,
        params.model_name,
        params.metric_name,
        params.quality,
        original_height,
        original_width,
        original_bitdepth,
        output_rows,
        output_cols,
        n_strings,
        length_strings,
        strings
    };

    save(finfo, output_file);
    
}


void decode_file(const std::string& compressed_file, const std::string& output_image_path, const std::string& model_dir) {
    fileInfo finfo = load(compressed_file);
    std::string model_name = finfo.model_name;
    std::string metric_name = finfo.metric_name;
    if (model_name != "bmshj2018-factorized" && model_name != "bmshj2018-factorized_relu") {
        std::cout << "----------model_name: " << model_name << std::endl;
        throw std::runtime_error("model is not supported");
    }

    std::string compressed_string = finfo.strings[0];
    char quality = finfo.quality;

    Params params = {
        finfo.quality,
        finfo.original_width,
        finfo.original_height,
        finfo.output_rows,
        finfo.output_cols,
        finfo.model_name,
        finfo.metric_name,
        nullptr,
        compressed_string,
    };

    decode_buffer(params, model_dir);
    std::cout << "----------params.original_height: " << params.original_height << std::endl;
    std::cout << "----------params.original_width: " << params.original_width << std::endl;

    cv::Mat output_image_mat = cv::Mat(params.original_height, params.original_width, CV_8UC3, params.rgb_data.get());
    cv::cvtColor(output_image_mat, output_image_mat, cv::COLOR_RGB2BGR);
    cv::imwrite(output_image_path, output_image_mat);
    std::cout << "Image saved to " << output_image_path << std::endl;
}

void decode_buffer(Params& params, const std::string& model_dir) {
    std::string compressed_string = params.compressed_string;
    uint32_t latent_rows = params.output_rows;
    uint32_t latent_cols = params.output_cols;
    uint32_t original_height = params.original_height;
    uint32_t original_width = params.original_width;
    uint32_t Scale = 16;
    std::string model_name = params.model_name;
    std::string metric_name = params.metric_name;
    char quality = params.quality;

    if (model_name != "bmshj2018-factorized" && model_name != "bmshj2018-factorized_relu") {
        std::cout << "----------model_name: " << model_name << std::endl;
        throw std::runtime_error("model is not supported");
    }
    
    // load entropy_bottleneck
    EntropyBottleNeck entropy_bottleneck_wrapper(model_dir + "/" + model_name + "-" + params.metric_name + "-q" + params.quality + "-entropy_bottleneck.npz");

    // load onnx runtime
    OnnxModelInferenceWrapper g_s(model_dir + "/" + model_name + "-" + params.metric_name + "-q" + params.quality + "-g_s.onnx", false);
    uint32_t C = static_cast<uint32_t>(g_s.inputDims_[1]);

    auto start_time = std::chrono::high_resolution_clock::now();
    // decompress
    std::vector<std::string> strings_list = {compressed_string};
    std::vector<int> input_shape = {latent_rows, latent_cols};

    auto start_time_decompress = std::chrono::high_resolution_clock::now();
    xt::xarray<float> decompressed_data = entropy_bottleneck_wrapper.decompress(strings_list, input_shape);
    auto end_time_decompress = std::chrono::high_resolution_clock::now();
    auto duration_decompress = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_decompress - start_time_decompress);
    std::cout << "decompress time taken: " << duration_decompress.count() << " milliseconds" << std::endl;

    // g_s
    uint32_t decompressed_data_height = latent_rows * Scale;
    uint32_t decompressed_data_width = latent_cols * Scale;
    std::vector<std::vector<float>> outputs_data_g_s = g_s.run(decompressed_data, 
                                                                {1, C, static_cast<int64_t>(latent_rows), static_cast<int64_t>(latent_cols)}, 
                                                                {1, 3, static_cast<int64_t>(decompressed_data_height), static_cast<int64_t>(decompressed_data_width)});
    auto start_time_decompress_post = std::chrono::high_resolution_clock::now();
                                                                
    std::vector<float> decompressed_data_float = outputs_data_g_s[0]; // N, C, H, W

    // decompress image
    std::vector<size_t> decompressed_data_shape{1, 3, static_cast<size_t>(decompressed_data_height), 
                                                    static_cast<size_t>(decompressed_data_width)};
    auto t = xt::adapt(decompressed_data_float.data(), decompressed_data_shape);
    auto t_crop = crop4d(t, original_height, original_width);
    auto decompressed_data_xarray = xt::eval(xt::view(t_crop, 0, xt::all(), xt::all(), xt::all()));
    std::cout << "----------decompressed_data_xarray.shape: " << xt::adapt(decompressed_data_xarray.shape()) << std::endl;

    //clamp_(0, 1)
    auto clamped_scaled = xt::eval(xt::clip(decompressed_data_xarray, 0.0, 1.0) * 255.0);
    std::cout << "----------clamped_scaled.shape: " << xt::adapt(clamped_scaled.shape()) << std::endl;
    auto transposed = xt::transpose(clamped_scaled, {1, 2, 0});
    std::cout << "----------transposed.shape: " << xt::adapt(transposed.shape()) << std::endl;
    auto final_uint8 = xt::cast<uint8_t>(transposed);
    auto final_uint8_eval = xt::eval(final_uint8);
    std::cout << "----------final_uint8_eval.shape: " << xt::adapt(final_uint8_eval.shape()) << std::endl;

    std::shared_ptr<uint8_t> buffer(
        reinterpret_cast<uint8_t*>(final_uint8_eval.data()),
        [final_uint8_eval](uint8_t*) mutable {
            // capture xarr by copy to hold its memory
            // do nothing on delete, because xarr manages memory
        }
    );
    params.rgb_data = buffer;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    auto end_time_decompress_post = std::chrono::high_resolution_clock::now();
    auto duration_decompress_post = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_decompress_post - start_time_decompress_post);
    std::cout << "decompress_post time taken: " << duration_decompress_post.count() << " milliseconds" << std::endl;
    std::cout << "decode time taken: " << duration.count() << " milliseconds" << std::endl;
}


void read_compressed_info(const std::string& compressed_file) {
    fileInfo finfo = load(compressed_file);
    std::cout << "model_name: " << finfo.model_name << std::endl;
    std::cout << "metric_name: " << finfo.metric_name << std::endl;
    std::cout << "quality: " << finfo.quality << std::endl;
    std::cout << "original_width: " << finfo.original_width << std::endl;
    std::cout << "original_height: " << finfo.original_height << std::endl;
    std::cout << "output_rows: " << finfo.output_rows << std::endl;
    std::cout << "output_cols: " << finfo.output_cols << std::endl;
    std::cout << "n_strings: " << finfo.n_strings << std::endl;
    std::cout << "length_strings: [";
    for (int i = 0; i < finfo.n_strings; i++) {
        std::cout << finfo.length_strings[i] << ", ";
    }
    std::cout << "]" << std::endl;
}