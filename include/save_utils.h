#pragma once
#include <iostream>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <string>



extern std::map<char, std::string> inverse_model_ids;

extern std::map<std::string, char> model_ids;

extern std::map<char, std::string> inverse_metric_ids;

extern std::map<std::string, char> metric_ids;

struct fileInfo {
    std::string filename;
    char model_id;
    char code;
    std::string model_name;
    std::string metric_name;
    char quality;
    uint32_t original_height;
    uint32_t original_width;
    uint32_t original_bitdepth;
    uint32_t output_rows;
    uint32_t output_cols;
    uint32_t n_strings;
    std::vector<uint32_t> length_strings;
    std::vector<std::string> strings;
};


uint32_t swap_uint32(uint32_t val);
uint32_t read_uint32(std::ifstream& file, int n = 1);
char read_uchar(std::ifstream& file, int n = 1);
std::string read_bytes(std::ifstream& file, int n);
void write_uint32(std::ofstream& file, uint32_t value, int n = 1);
void write_uchar(std::ofstream& file, char value, int n = 1);
void write_bytes(std::ofstream& file, const std::string& value);
fileInfo load(const std::string& filename);
void save(const fileInfo& info, const std::string& output_path);
void build_code(char metric, char quality, char& code);
void parse_code(char code, char& quality, char& metric);

