#include <map>
#include <fstream>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <functional>
#include <thread>
#include "onnx_model_wrapper.h"
#include <filesystem>
namespace fs = std::filesystem;

OnnxModelInferenceWrapper::OnnxModelInferenceWrapper(const std::string& modelFilepath, bool useOPENVINO) 
    : env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, modelFilepath.c_str()),
      sessionOptions_(),  // 默认构造
      session_(nullptr),  // 先初始化为nullptr
      allocator_(),
      numInputNodes_(0),
      numOutputNodes_(0),
      inputType_(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED),
      outputType_(ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED)
{

    // 不存在
    if (!fs::exists(modelFilepath)) {
        throw std::runtime_error("modelFilepath " + modelFilepath + " not found");
    }

    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    unsigned int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1; // fallback
    if (num_threads > 8) num_threads = 8;
    if (num_threads > 16) num_threads = 16;
    std::cout << "----------num_threads: " << num_threads << std::endl;
    sessionOptions_.SetIntraOpNumThreads(num_threads);
    sessionOptions_.SetInterOpNumThreads(1);
    sessionOptions_.EnableMemPattern();
    sessionOptions_.EnableCpuMemArena();
    // sessionOptions_.SetOptimizedModelFilePath(std::string(modelFilepath + ".optimized").c_str());


    //Appending OpenVINO Execution Provider API
    if (useOPENVINO) {
        // Using OPENVINO backend
        OrtOpenVINOProviderOptions options;
        options.device_type = "CPU_FP32"; //Other options are: GPU_FP32, GPU_FP16, MYRIAD_FP16
        std::cout << "OpenVINO device type is set to: " << options.device_type << std::endl;
        sessionOptions_.AppendExecutionProvider_OpenVINO(options);
    }
    
    //Creation: The Ort::Session is created here
    session_ = Ort::Session(env_, modelFilepath.c_str(), sessionOptions_);

    allocator_ = Ort::AllocatorWithDefaultOptions();

    numInputNodes_ = session_.GetInputCount();
    numOutputNodes_ = session_.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes_ << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes_ << std::endl;

    auto inputNodeName = session_.GetInputNameAllocated(0, allocator_);
    inputName_ = std::string(inputNodeName.get());
    std::cout << "Input Name: " << inputName_ << std::endl;

    Ort::TypeInfo inputTypeInfo = session_.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    inputType_ = inputTensorInfo.GetElementType();
    std::cout << "Input Type: " << inputType_ << std::endl;

    inputDims_ = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims_ << std::endl;

    auto outputNodeName = session_.GetOutputNameAllocated(0, allocator_);
    outputName_ = std::string(outputNodeName.get());
    std::cout << "Output Name: " << outputName_ << std::endl;

    Ort::TypeInfo outputTypeInfo = session_.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    outputType_ = outputTensorInfo.GetElementType();
    std::cout << "Output Type: " << outputType_ << std::endl;

    outputDims_ = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims_    << std::endl;

}


std::vector<std::vector<float>> OnnxModelInferenceWrapper::run(const xt::xarray<float>& input, const std::vector<int64_t>& inputDims, const std::vector<int64_t>& outputDims) {
    auto tic = std::chrono::high_resolution_clock::now();

    //Run Inference

    /* To run inference using ONNX Runtime, the user is responsible for creating and managing the 
    input and output buffers. These buffers could be created and managed via std::vector.
    The linear-format input data should be copied to the buffer for ONNX Runtime inference. */


    int inputTensorSize = vectorProduct(inputDims);
    // std::vector<float> inputTensorValues(input.begin(), input.end());
    // std::cout << "Input Tensor Values: " << inputTensorValues.size() << std::endl;

    int outputTensorSize = vectorProduct(outputDims);
    
    // 根据输出类型创建正确的tensor
    std::vector<float> outputTensorValues;
    outputTensorValues.resize(outputTensorSize);


    /* Once the buffers were created, they would be used for creating instances of Ort::Value 
    which is the tensor format for ONNX Runtime. There could be multiple inputs for a neural network, 
    so we have to prepare an array of Ort::Value instances for inputs and outputs respectively even if 
    we only have one input and one output. */

    std::vector<const char*> inputNames{inputName_.c_str()};
    std::vector<const char*> outputNames{outputName_.c_str()};

    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    /*
    Creating ONNX Runtime inference sessions, querying input and output names, 
    dimensions, and types are trivial.
    Setup inputs & outputs: The input & output tensors are created here. */

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    
    std::cout << "memoryInfo created"  << std::endl;

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(input.data()), inputTensorSize, inputDims.data(),
        inputDims.size()));
    
    std::cout << "inputTensors: " << inputTensors.size() << std::endl;

    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
            outputDims.data(), outputDims.size()));

    /* To run inference, we provide the run options, an array of input names corresponding to the 
    inputs in the input tensor, an array of input tensor, number of inputs, an array of output names 
    corresponding to the the outputs in the output tensor, an array of output tensor, number of outputs. */
    session_.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    std::vector<std::vector<float>> outputTensors_list;
    uint32_t output_size = outputTensors.size();
    for (uint32_t i = 0; i < output_size; i++) {
        outputTensors_list.push_back(outputTensorValues);
    }

    auto toc = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);
    std::cout << "OnnxModelInferenceWrapper::run time taken: " << duration.count() << " milliseconds" << std::endl;

    return outputTensors_list;
}

OnnxModelInferenceWrapper::~OnnxModelInferenceWrapper() {
    // 析构函数实现
    // Ort::Session 和 Ort::Env 会自动清理
}



