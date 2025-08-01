#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <map>
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xnpy.hpp>
#include <xtensor/misc/xpad.hpp>

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}

class OnnxModelInferenceWrapper {
    public:
        OnnxModelInferenceWrapper(const std::string& onnx_path, bool useOPENVINO);
        ~OnnxModelInferenceWrapper();

        std::vector<std::vector<float>> run(const xt::xarray<float>& input, const std::vector<int64_t>& inputDims, const std::vector<int64_t>& outputDims);

        std::vector<int64_t> inputDims_;
        std::vector<int64_t> outputDims_;
        
    private:
        Ort::Env env_;
        Ort::SessionOptions sessionOptions_;
        Ort::Session session_;
        Ort::AllocatorWithDefaultOptions allocator_;
        size_t numInputNodes_;
        size_t numOutputNodes_;
        std::string inputName_;
        std::string outputName_;
        ONNXTensorElementDataType inputType_;
        ONNXTensorElementDataType outputType_;
};