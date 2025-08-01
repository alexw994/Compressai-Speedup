#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include "save_utils.h"

// 字节序转换函数（大端转小端或小端转大端）
uint32_t swap_uint32(uint32_t val) {
    return ((val >> 24) & 0xff) |       // 移到最低字节
           ((val << 8) & 0xff0000) |    // 移到第3字节
           ((val >> 8) & 0xff00) |      // 移到第2字节
           ((val << 24) & 0xff000000);  // 移到最高字节
}


std::map<char, std::string> inverse_model_ids = {
    {0, "bmshj2018-factorized"}, 
    {1, "bmshj2018-factorized-relu"}, 
    {2, "bmshj2018-hyperprior"}, 
    {3, "mbt2018-mean"},
    {4, "mbt2018"}, 
    {5, "cheng2020-anchor"}, 
    {6, "cheng2020-attn"}, 
    {7, "bmshj2018-hyperprior-vbr"}, 
    {8, "mbt2018-mean-vbr"}, 
    {9, "mbt2018-vbr"}, 
    {10, "hrtzxf2022-pcc-rec"}, 
    {11, "sfu2023-pcc-rec-pointnet"}, 
    {12, "sfu2024-pcc-rec-pointnet2-ssg"}, 
    {13, "ssf2020"}
};

std::map<std::string, char> model_ids = {
    {"bmshj2018-factorized", 0},
    {"bmshj2018-factorized-relu", 1},
    {"bmshj2018-hyperprior", 2},
    {"mbt2018-mean", 3},
    {"mbt2018", 4},
    {"cheng2020-anchor", 5},
    {"cheng2020-attn", 6},
    {"bmshj2018-hyperprior-vbr", 7},
    {"mbt2018-mean-vbr", 8},
    {"mbt2018-vbr", 9},
    {"hrtzxf2022-pcc-rec", 10},
    {"sfu2023-pcc-rec-pointnet", 11},
    {"sfu2024-pcc-rec-pointnet2-ssg", 12},
    {"ssf2020", 13}
};

std::map<char, std::string> inverse_metric_ids = {
    {0, "mse"},
    {1, "ms-ssim"}
};

std::map<std::string, char> metric_ids = {
    {"mse", 0},
    {"ms-ssim", 1}
};


uint32_t read_uint32(std::ifstream& file, int n) {
    uint32_t buf;
    file.read(reinterpret_cast<char*>(&buf), n * sizeof(uint32_t));
    return swap_uint32(buf);
}

char read_uchar(std::ifstream& file, int n) {
    char buf;
    file.read(reinterpret_cast<char*>(&buf), n * sizeof(char));
    return buf;
}

std::string read_bytes(std::ifstream& file, int n) {
    char* buf = new char[n];
    file.read(buf, n * sizeof(char));
    std::string value(buf, n);
    delete[] buf;
    return value;
}

void write_uint32(std::ofstream& file, uint32_t value, int n) {
    uint32_t buf = swap_uint32(value);
    file.write(reinterpret_cast<char*>(&buf), n * sizeof(uint32_t));
}

void write_uchar(std::ofstream& file, char value, int n) {
    file.write(reinterpret_cast<char*>(&value), n * sizeof(char));
}


void write_bytes(std::ofstream& file, const std::string& value) {
    file.write(value.c_str(), value.length());
}


void parse_code(char code, char& quality, char& metric) {
    quality = (code & 0x0F) + 1;
    metric = code >> 4;
}

void build_code(char metric, char quality, char& code) {
    code = (metric << 4) | ((quality - 1) & 0x0F);
    std::cout << "    metric: " << static_cast<int>(metric) << " code: " << static_cast<int>(code) << std::endl;
}

fileInfo load(const std::string& filename) {
    // 读取文件
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return fileInfo();
    }

    // 读取header
    char model_id = read_uchar(file);
    char code = read_uchar(file);
    char quality, metric;
    parse_code(code, quality, metric);
    uint32_t original_height = read_uint32(file);
    uint32_t original_width = read_uint32(file);
    uint32_t original_bitdepth = read_uchar(file);
    uint32_t output_rows = read_uint32(file);
    uint32_t output_cols = read_uint32(file);
    uint32_t n_strings = read_uint32(file);

    std::string model_name = inverse_model_ids[model_id];
    std::string metric_name = inverse_metric_ids[metric];

    std::cout << "model_id: " << static_cast<int>(model_id) << std::endl;
    std::cout << "code: " << static_cast<int>(code) << std::endl;
    std::cout << "quality: " << static_cast<int>(quality) << std::endl;
    std::cout << "metric: " << static_cast<int>(metric) << std::endl;
    std::cout << "model_name: " << model_name << std::endl;
    std::cout << "metric_name: " << metric_name << std::endl;
    std::cout << "original_height: " << original_height << std::endl;
    std::cout << "original_width: " << original_width << std::endl;
    std::cout << "output_rows: " << output_rows << std::endl;
    std::cout << "output_cols: " << output_cols << std::endl;
    std::cout << "original_bitdepth: " << original_bitdepth << std::endl;
    std::cout << "n_strings: " << n_strings << std::endl;

    std::vector<std::string> strings;
    std::vector<uint32_t> length_strings;
    for (size_t i = 0; i < n_strings; i++) {
        uint32_t length = read_uint32(file);
        std::string string = read_bytes(file, length);
        length_strings.push_back(length);
        strings.push_back(string);
    }
    std::cout << "strings length: [";
    for (size_t i = 0; i < n_strings; i++) {
        std::cout << length_strings[i] << ",";
    }
    std::cout << "]" << std::endl;

    file.close();

    fileInfo info = {
        filename,
        model_id,
        code,
        model_name,
        metric_name,
        quality,
        original_height,
        original_width,
        original_bitdepth,
        output_rows,
        output_cols,
        n_strings,
        length_strings,
        strings
    };

    return info;
}


void save(const fileInfo& info, const std::string& output_path) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << output_path << std::endl;
        return;
    }

    std::cout << "info.model_id: " << static_cast<int>(info.model_id) << std::endl;
    std::cout << "info.code: " << static_cast<int>(info.code) << std::endl;
    std::cout << "info.original_height: " << info.original_height << std::endl;
    std::cout << "info.original_width: " << info.original_width << std::endl;
    std::cout << "info.original_bitdepth: " << static_cast<int>(info.original_bitdepth) << std::endl;
    std::cout << "info.output_rows: " << info.output_rows << std::endl;
    std::cout << "info.output_cols: " << info.output_cols << std::endl;
    std::cout << "info.n_strings: " << info.n_strings << std::endl;
    std::cout << "info.length_strings: [";
    for (size_t i = 0; i < info.n_strings; i++) {
        std::cout << info.length_strings[i] << ",";
    }
    std::cout << "]" << std::endl;

    write_uchar(file, info.model_id);
    write_uchar(file, info.code);
    write_uint32(file, info.original_height);
    write_uint32(file, info.original_width);
    write_uchar(file, info.original_bitdepth);
    write_uint32(file, info.output_rows);
    write_uint32(file, info.output_cols);
    write_uint32(file, info.n_strings);
    for (size_t i = 0; i < info.n_strings; i++) {
        write_uint32(file, info.length_strings[i]);
        write_bytes(file, info.strings[i]);
    }

    file.close();

    std::cout << "save file: " << output_path << std::endl;
}