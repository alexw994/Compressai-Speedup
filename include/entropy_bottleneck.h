#include <onnxruntime_cxx_api.h>
#include <cnpy.h>
#include "rans_interface.hpp"
#include <xtensor/containers/xarray.hpp>
#include <xtensor/io/xio.hpp>
#include <xtensor/views/xview.hpp>
#include <xtensor/io/xnpy.hpp>
#include <variant>

class EntropyBottleNeck {
    public:
        EntropyBottleNeck(const std::string& npz_path);
        ~EntropyBottleNeck();

        std::vector<std::string> compress(const xt::xarray<float>& input);
        xt::xarray<float> decompress(const std::vector<std::string>& strings_list, const std::vector<int>& input_shape);

        RansEncoder rans_enc = RansEncoder();
        RansDecoder rans_dec = RansDecoder();
    
    private:
        std::vector<std::vector<int>> quantized_cdf_;
        std::vector<int> cdf_length_;
        std::vector<int> offset_;
        std::vector<float> quantiles_;
};
