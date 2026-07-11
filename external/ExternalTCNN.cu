/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "tiny-cuda-nn/common.h"
#include "tiny-cuda-nn/network.h"
#include "tiny-cuda-nn/rtc_kernel.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#define DLL_MACRO_NEED_EXPORT
#include "ExternalTCNN.hpp"

namespace
{

template<auto FuncPtr>
struct Deleter
{
    template<typename T>
    void operator()(T resource) const
    {
        FuncPtr(resource);
    }
};

template<typename T>
auto LoadParams(const std::filesystem::path &encodingPath, std::size_t size)
{
    auto byteSize = size * sizeof(T);
    T *gpuBasePtr;
    if (cudaMalloc(&gpuBasePtr, byteSize) != cudaSuccess)
        throw std::runtime_error{ "CUDA allocation failure" };
    std::unique_ptr<T, Deleter<cudaFree>> params{ gpuBasePtr };

    auto baseArr = std::make_unique<T[]>(size);
    std::ifstream fin{ encodingPath, std::ios::binary };
    fin.exceptions(std::ios::badbit | std::ios::failbit);
    fin.read(reinterpret_cast<char *>(baseArr.get()), byteSize);
    if (cudaMemcpy(gpuBasePtr, baseArr.get(), byteSize,
                   cudaMemcpyHostToDevice) != cudaSuccess)
        throw std::runtime_error{ "CUDA copy failure" };

    return params;
}

int GetPaddedInputWidth(int width)
{
    return tcnn::next_multiple(width, 16);
}

} // namespace

struct TCNNHandler
{
    struct Network
    {
        Network(tcnn::Network<half> *ptr) : network{ ptr } {}

        std::unique_ptr<tcnn::Network<half>> network;
        std::unique_ptr<half, Deleter<cudaFree>> params;
    };

    std::vector<Network> networks;
    std::unique_ptr<half *, Deleter<cudaFree>> paramsDeviceVector;
    tcnn::CudaRtcKernel fusedKernel;
};

static void *NewTCNNHandlerImpl(const char *workDirRaw, int featureDim,
                                int splitOffset, int albedoDim)
{
    std::filesystem::path workDir{ workDirRaw };
    std::ifstream fin{ workDir / "config.json" };
    auto config = nlohmann::json::parse(fin);

    std::vector<TCNNHandler::Network> networks;
    auto EmplaceNetwork = [&](const std::filesystem::path &paramPath,
                              const nlohmann::json &networkConfig) {
        auto &newNetwork =
            networks.emplace_back(tcnn::create_network<half>(networkConfig));

        newNetwork.params =
            LoadParams<half>(paramPath, newNetwork.network->n_params());
        newNetwork.network->set_params(newNetwork.params.get(),
                                       newNetwork.params.get(), nullptr);
        newNetwork.network->convert_params_to_jit_layout(nullptr, true);
    };

    int networkNum = config["step_num"];
    if (networkNum <= 0)
        throw std::runtime_error{ "Step number is abnormal." };

    auto &networkConfig = config["network"];
    auto inputDim = featureDim - albedoDim * 2;
    auto paddedInputDim = GetPaddedInputWidth(inputDim);
    networkConfig["n_input_dims"] = paddedInputDim;
    networkConfig["n_output_dims"] = inputDim;
    networkConfig.erase("output_activation");

    for (int i = 0; i < networkNum; i++)
    {
        EmplaceNetwork(workDir / fmt::format("tcnn.mlp{}.bin", i),
                       networkConfig);
    }
    auto &finalNetworkConfig = config["network_final"];
    finalNetworkConfig["n_input_dims"] = paddedInputDim;
    finalNetworkConfig["n_output_dims"] = 1;
    EmplaceNetwork(workDir / "tcnn.final_mlp.bin", finalNetworkConfig);

    // Then generate fused kernel.
    fin.close();
    fin.open(KERNEL_PATH);
    std::string buffer;
    {
        std::ostringstream str;
        fin >> str.rdbuf();
        buffer = str.str();
    }

    std::string prepareLayerCode =
        networks.front().network->generate_device_function(
            "PrepareLayerInference");
    std::string finalLayerCode =
        networks.back().network->generate_device_function(
            "FinalLayerInference");

    std::string kernelCode = fmt::format(
        fmt::runtime(buffer), fmt::arg("FEATURE_DIM", featureDim),
        fmt::arg("SPLIT_OFFSET", splitOffset),
        fmt::arg("ALBEDO_DIM", albedoDim), fmt::arg("INPUT_DIM", inputDim),
        fmt::arg("EXTRA_DIM", paddedInputDim - inputDim),
        fmt::arg("PREPARE_LAYER_INFERENCE", prepareLayerCode),
        fmt::arg("FINAL_LAYER_INFERENCE", finalLayerCode),
        fmt::arg("NETWORK_NUM", networks.size() - 1));

    // std::ofstream{ "debug.cu" } << kernelCode;

    half **paramsDeviceVectorRaw;
    if (cudaMalloc(&paramsDeviceVectorRaw, (networkNum + 1) * sizeof(half *)) !=
        cudaSuccess)
    {
        throw std::runtime_error{ "CUDA allocation failure" };
    }

    std::unique_ptr<half *, Deleter<cudaFree>> paramsDeviceVector{
        paramsDeviceVectorRaw
    };

    std::vector<half *> params;
    for (const auto &net : networks)
        params.push_back(net.network->inference_params());
    std::cout << params.size() << "\n";

    if (cudaMemcpy(paramsDeviceVectorRaw, params.data(),
                   params.size() * sizeof(half *),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        throw std::runtime_error{ "Constant memory copy copy fails." };
    }

    return new TCNNHandler{
        std::move(networks),
        std::move(paramsDeviceVector),
        tcnn::CudaRtcKernel{ "RadianceInference", kernelCode },
    };
}

extern "C" DLL_EXPORT void *NewTCNNHandler(const char *workDirRaw,
                                           int featureDim, int splitOffset,
                                           int albedoDim)
{
    try
    {
        return NewTCNNHandlerImpl(workDirRaw, featureDim, splitOffset,
                                  albedoDim);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << "\n";
    }
    // Don't allow invalid handle to return.
    std::terminate();
}

static TCNNHandler *GetHandle(void *handle)
{
    return static_cast<TCNNHandler *>(handle);
}

extern "C" DLL_EXPORT void ExternalTCNNInference(
    void *handleRaw, void *featureBuffer, void *radianceBuffer,
    void *positionBuffer, std::size_t size, std::uint32_t albedoSkipHint,
    Color3 scale)
{
    assert(size % WARP_SIZE == 0);
    auto handle = GetHandle(handleRaw);

    uint32_t threads = 128;
    uint32_t blocks = tcnn::div_round_up<std::size_t>(size, threads);
    // Position buffer is actually glm::vec4, leading such access UB.
    // But practically it's fine.
    handle->fusedKernel.launch(
        blocks, threads, 0, nullptr, static_cast<half *>(featureBuffer),
        static_cast<half *>(radianceBuffer),
        static_cast<float *>(positionBuffer), size, albedoSkipHint,
        tcnn::vec3{ scale.r, scale.g, scale.b },
        handle->paramsDeviceVector.get());

    return;
}

extern "C" DLL_EXPORT void DeleteTCNNHandler(void *handle)
{
    delete GetHandle(handle);
}