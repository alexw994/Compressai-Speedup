#include <iostream>
#include <string>
#include "codec.h"


void print_help(char* argv[]) {
    std::cerr << "-----------Only Support bmshj2018-factorized Model-----------" << std::endl;
    std::cerr << "-----------Default Model Path: ./models/bmshj2018-factorized-mse-q3-g_a.onnx-----------" << std::endl;
    std::cerr << "-----------set env AICODEC_MODEL_DIR to set model_dir-----------" << std::endl;
    std::cerr << "Usage: " << argv[0] << " encode <image_path> <output_file>" << std::endl;
    std::cerr << "Example: " << argv[0] << " encode /path/to/image.jpg /path/to/output.cmpai" << std::endl;
    std::cerr << "--------------------------------" << std::endl;
    std::cerr << "Usage: " << argv[0] << " decode <compressed_file> <output_image_path>" << std::endl;
    std::cerr << "Example: " << argv[0] << " decode /path/to/compressed.cmpai /path/to/output.jpg" << std::endl;
    std::cerr << "--------------------------------" << std::endl;   
}


int main(int argc, char* argv[])

{
    // encode <image_path> <output_file> <model_name> <metric_name> <quality>
    // decode <compressed_file> <output_image_path>
    if (argc == 1 || argv[1] == "help") {
        print_help(argv);
        return 1;
    }

    const std::string& mode = argv[1];

    if (mode == "encode") {
        if (argc != 4) {
            print_help(argv);
            return 1;
        }

        const std::string& image_path = argv[2];
        const std::string& output_file = argv[3];
        const std::string model_name = "bmshj2018-factorized";
        const std::string metric_name = "mse";
        const char quality = '3';
        const std::string model_dir = std::getenv("AICODEC_MODEL_DIR") ? std::getenv("AICODEC_MODEL_DIR") : "./models";
        Params params = {
            quality,
            0,
            0,
            0,
            0,
            model_name,
            metric_name,
            nullptr,
            "",
        };
        encode_file(image_path, output_file, params, model_dir);
    } else if (mode == "decode") {  
        if (argc != 4) {
            print_help(argv);
            return 1;
        }

        const std::string& compressed_file = argv[2];
        const std::string& output_image_path = argv[3];
        const std::string model_dir = std::getenv("AICODEC_MODEL_DIR") ? std::getenv("AICODEC_MODEL_DIR") : "./models";
        decode_file(compressed_file, output_image_path, model_dir);
    } else {
        print_help(argv);
        return 1;
    } 

    return 0;
}
