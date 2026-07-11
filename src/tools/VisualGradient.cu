/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "render/Network.hpp"
#include "stbi/stb_image_write.h"
#include "utils/Utils.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

using FloatType = half;

#define MAX_FEATURE_DIM 8

__global__ void GetEncodingGradient(EncodingGPUProxy<FloatType> encoding,
                                    float *dstBuffer, std::size_t width,
                                    std::size_t height, int levelNum,
                                    int alongAxis)
{
    constexpr int Dim = 2;
    auto pixelX = blockIdx.x * blockDim.x + threadIdx.x,
         pixelY = blockIdx.y * blockDim.y + threadIdx.y;
    if (pixelX >= width || pixelY >= height)
        return;

    auto featureDim = encoding.GetFeatureDim();
    assert(alongAxis < Dim && featureDim <= MAX_FEATURE_DIM);

    FloatType last[MAX_FEATURE_DIM], next[MAX_FEATURE_DIM];

    glm::vec2 currCoord{ float(pixelX) / width, float(pixelY) / height },
        deltaCoord{ 1.0f / width, 1.0f / height };
    glm::vec2 lastCoord = currCoord, nextCoord = currCoord;
    lastCoord[alongAxis] -= deltaCoord[alongAxis];
    nextCoord[alongAxis] += deltaCoord[alongAxis];

    std::size_t size = width * height, offset = pixelY * width + pixelX;
    for (int i = 0; i < levelNum; i++)
    {
        auto ptr = dstBuffer + size * i + offset;
        encoding.Sample<Dim>(last, featureDim, i, lastCoord);
        encoding.Sample<Dim>(next, featureDim, i, nextCoord);

        float total = 0;
        for (int j = 0; j < featureDim; j++)
        {
            float diff = float(next[j] - last[j]) / (2 * deltaCoord[alongAxis]);
            total += diff * diff;
        }
        *ptr = sqrt(total) / 40.0f;
    }
}

int main(int argc, const char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: ViewAdjustment CONFIG_ROOT_DIR";
        return 0;
    }

    std::filesystem::path rootPath{ /* SPECIFY PATH */ };
    std::ifstream fin{ rootPath / "config.json" };
    auto config = nlohmann::json::parse(fin);

    auto encoding = Encoding<FloatType>{ rootPath / "albedo.bin",
                                         config["hg_encoding"], 2 };
    const auto &resolution = encoding.GetResolution();
    auto finalLayerRes = resolution.back();
    int layerNum = resolution.size();

    auto singleSize = finalLayerRes * finalLayerRes;
    UniqueResource<float *, CUDADeleter> buffer;
    float *rawPtr;
    CheckCUDAError(
        cudaMallocManaged(&rawPtr, singleSize * layerNum * sizeof(float)));
    buffer.Reloc(rawPtr);

    dim3 dimBlock(8, 4);
    dim3 dimGrid(RoundUpDiv(finalLayerRes, dimBlock.x),
                 RoundUpDiv(finalLayerRes, dimBlock.y));
    // cos_theta is the second axis.
    GetEncodingGradient<<<dimGrid, dimBlock>>>(
        encoding.ToProxy(), rawPtr, finalLayerRes, finalLayerRes, layerNum, 1);

    cudaDeviceSynchronize();

    for (int i = 0; i < layerNum; i++)
    {
        auto ptr = rawPtr + i * singleSize;
        auto path = "layer" + std::to_string(i) + ".hdr";
        stbi_write_hdr(path.c_str(), finalLayerRes, finalLayerRes, 1, ptr);
    }

    return 0;
}