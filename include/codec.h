#pragma once
#include <string>
#include <memory>
#include <vector>

struct Params {
    char quality;
    uint32_t original_width;
    uint32_t original_height;
    uint32_t output_rows;
    uint32_t output_cols;
    std::string model_name;
    std::string metric_name;
    std::shared_ptr<uint8_t> rgb_data;
    std::string compressed_string;
};


void encode_buffer(Params& params, const std::string& model_dir);
void encode_file(const std::string& input_file, const std::string& output_file, Params& params, const std::string& model_dir);
void decode_buffer(Params& params, const std::string& model_dir);
void decode_file(const std::string& compressed_file, const std::string& output_image_path, const std::string& model_dir);
void read_compressed_info(const std::string& compressed_file);