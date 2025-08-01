# CompressAI Speedup Codec

使用[CompressAI](https://github.com/InterDigitalInc/CompressAI)框架实现高性能图像压缩, 使用c++加速。

## 功能特性

高性能图像压缩和解压缩，**仅支持bmshj2018-factorized mse q=3模型**

## 快速开始
### 直接使用
下载release中的预编译包，按照示例直接使用

```bash
# 压缩图像
./bin/cmpai-cli encode input.jpg output.cmpai

# 解压缩图像
./bin/cmpai-cli decode output.cmpai output.jpg
```

可以使用环境变量AICODEC_MODEL_DIR指定模型路径

保存的.cmpai文件格式和[CompressAI](https://github.com/InterDigitalInc/CompressAI)项目导出的压缩文件保持一致，可以互相读写

### 编译安装

```bash
# 克隆项目
git clone https://github.com/alexw994/Compressai-Speedup.git
cd Compressai-Speedup

# 创建构建目录
mkdir build && cd build

# 配置和编译
cmake ..
make -j$(nproc)

# 安装（可选）
make install
```
