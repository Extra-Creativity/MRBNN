/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "RenderInterface.hpp"
#include "EnvironmentBaking.hpp"
#include "Network.hpp"
#include "RenderPayload.hpp"

#define DLL_MACRO_NEED_IMPORT
#include "ExternalTCNN.hpp"

#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <random>

struct RenderInterface::NetworkData
{
    Encoding<NetworkFloat> baseEncoding;

    Encoding<NetworkFloat> viewEncoding;
    Encoding<NetworkFloat> lightEncoding;
    Encoding<NetworkFloat> hgEncoding;
    Encoding<NetworkFloat> albedoEncoding;

    SphericalEncoding<NetworkFloat, 2> shEncoding;
    std::vector<Network<NetworkFloat>> networks;

    NetworkGlobalPayload GetPayload() const
    {
        NetworkGlobalPayload payload{
            baseEncoding.ToProxy(),   viewEncoding.ToProxy(),
            lightEncoding.ToProxy(),  hgEncoding.ToProxy(),
            albedoEncoding.ToProxy(), shEncoding.ToProxy(),
        };
        payload.networkNum = static_cast<int>(networks.size());
        CheckError(
            payload.networkNum <= NetworkGlobalPayload::s_maxNetworkNum,
            LAZY_STR(std::format("At most {} networks are supported.",
                                 NetworkGlobalPayload::s_maxNetworkNum)));

        for (int i = 0; i < payload.networkNum; i++)
        {
            payload.networks[i] = networks[i].ToProxy();
        }
        payload.viewEncDim = viewEncoding.GetInputDims();
        return payload;
    }
};

struct RenderInterface::AuxData
{
    UniqueResource<float *, CUDADeleter> lastBuffer, newBuffer;
    unsigned int resx = 0, resy = 0, channelNum = 0;

    RenderParametersPayload::CameraInfo lastCamera;
    EnvironmentBaking envBaking;

    class UpdateGuard
    {
        AuxData *ptr_;
        unsigned int resx_, resy_, channelNum_;
        bool refreshLastBuffer_;

    public:
        UpdateGuard(AuxData *ptr, unsigned int resx, unsigned int resy,
                    unsigned int channelNum, bool refreshLastBuffer) noexcept
            : ptr_{ ptr }, resx_{ resx }, resy_{ resy },
              channelNum_{ channelNum }, refreshLastBuffer_{ refreshLastBuffer }
        {
        }
        ~UpdateGuard()
        {
            ptr_->resx = resx_, ptr_->resy = resy_,
            ptr_->channelNum = channelNum_;

            if (refreshLastBuffer_)
                ptr_->lastBuffer.Reset();
        }
    };

    // This guard only updates buffers.
    [[nodiscard]] UpdateGuard SwapBuffers(unsigned int newResX,
                                          unsigned int newResY,
                                          unsigned int newChannelNum)
    {
        // When entering this function, `lastBuffer` stores frame n-2,
        // and `newBuffer` stores frame n-1. When they have the same
        // size (`lastSize`), then both of them are not nullptr.
        // Thus, when `lastSize == newSize`, just swap buffers to make
        // lastBuffer stores frame n-1 and newBuffer ready for frame n.
        auto lastSize = resx * resy * channelNum,
             newSize = newResX * newResY * newChannelNum;
        // In most cases, we won't resize buffer.
        if (lastSize == newSize && lastBuffer.Get() != nullptr) [[likely]]
        {
            lastBuffer.Swap(newBuffer);
            return { this, newResX, newResY, newChannelNum, false };
        }

        // Otherwise, two cases:
        // 1. lastSize != newSize, then in the next frame, frame n-2 will have
        //    different size. We thus deallocate frame n-2 to let frame n
        //    reallocate in the next frame.
        // 2. lastSize == newSize, but lastBuffer is nullptr. This is next frame
        //    after case 1, so we reallocate.
        float *newBufferPtr;
        CheckCUDAError(cudaMalloc(&newBufferPtr, newSize * sizeof(float)));

        lastBuffer = std::move(newBuffer);
        newBuffer.Reloc(newBufferPtr);
        return { this, newResX, newResY, newChannelNum, lastSize != newSize };
    }
};

RenderInterface::RenderInterface(RenderInterface &&) noexcept = default;
RenderInterface &RenderInterface::operator=(RenderInterface &&) noexcept =
    default;
RenderInterface::~RenderInterface() = default;

RenderInterface::RenderInterface(const std::filesystem::path &workDir)
    : auxData_{ new AuxData{} }
{
    std::ifstream fin{ workDir / "config.json" };
    auto config = nlohmann::json::parse(fin);

    auto volumeInfo = config["volume"];
    {
        auto GetResolution = [](auto &resInfo) {
            std::array<int, 3> resolution;
            for (int i = 0; i < 3; i++)
                resolution[i] = resInfo[i].get<int>();
            return resolution;
        };

        auto volumePath = volumeInfo["path"].get_ref<std::string &>();
        auto pathPtr = reinterpret_cast<const char8_t *>(volumePath.data());
        std::filesystem::path path{ pathPtr, pathPtr + volumePath.size() };
        if (path.is_relative())
            path = std::filesystem::path{ RELA_PATH_ROOT } / path;

        std::array<int, 3> resolution = GetResolution(volumeInfo["resolution"]);
        std::size_t skipByteNum =
            volumeInfo.value("skip_byte_num", std::size_t{ 0 });
        int mipmapLevel = volumeInfo.value("mipmap_level", 0);
        Bounds3f bound{ { -0.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f } };

        if (auto it = volumeInfo.find("bound"); it != volumeInfo.end())
        {
            auto &arr = it.value().get_ref<nlohmann::json::array_t &>();
            if (arr.size() != 2)
                throw std::invalid_argument{ "Incorrect bound specification." };

            glm::vec3 corners[2];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++)
                    corners[i][j] = arr[i][j];
            if (!glm::all(glm::lessThan(corners[0], corners[1])))
            {
                throw std::runtime_error{
                    "Bound min corner must be less than max corner."
                };
            }

            bound = Bounds3f{ corners[0], corners[1] };
        }

        if (volumeInfo.contains("albedo"))
        {
            std::cout << "Reading albedo grid...\n";
            auto &albedoInfo = volumeInfo["albedo"];
            auto albedoPathStr = albedoInfo["path"].get_ref<std::string &>();
            auto albedoPathPtr =
                reinterpret_cast<const char8_t *>(albedoPathStr.data());
            std::filesystem::path albedoPath{
                albedoPathPtr, albedoPathPtr + albedoPathStr.size()
            };

            if (albedoPath.is_relative())
            {
                albedoPath =
                    std::filesystem::path{ RELA_PATH_ROOT } / albedoPath;
            }

            AlbedoVolume albedo{
                albedoPath, albedoInfo.value("skip_byte_num", std::size_t{ 0 }),
                GetResolution(albedoInfo["resolution"]),
                albedoInfo.value("channel_num", 3)
            };
            volume_ = GPUSimpleVolume{ path,  skipByteNum, resolution,
                                       bound, mipmapLevel, std::move(albedo) };
        }
        else
        {
            volume_ = GPUSimpleVolume{ path, skipByteNum, resolution, bound,
                                       mipmapLevel };
        }
    }

    if (auto mapConfig = config["tr_map"]; mapConfig.contains("begin_level"))
    {
        transmittanceMap_ = TransmittanceMap{
            *volume_.Get(),
            mapConfig["begin_level"].get<int>(),
            mapConfig["mipmap_level"].get<int>(),
        };
    }
    else
    {
        auto resolution = mapConfig["resolution"];
        transmittanceMap_ = TransmittanceMap{
            std::array<int, 3>{ resolution[0], resolution[1], resolution[2] },
            mapConfig["mipmap_level"].get<int>(),
        };
    }

    int viewEncDim = config.value("view_enc_dim", 2);
    networkData_.reset(new NetworkData{
        Encoding<NetworkFloat>{ workDir / "base.bin", config["volume_encoding"],
                                3 },
        Encoding<NetworkFloat>{ workDir / "view.bin", config["view_encoding"],
                                viewEncDim },
        Encoding<NetworkFloat>{ workDir / "light.bin", config["light_encoding"],
                                2 },
        Encoding<NetworkFloat>{ workDir / "hg.bin", config["hg_encoding"], 2 },
        Encoding<NetworkFloat>{ workDir / "albedo.bin",
                                config["albedo_encoding"], 2 },
        SphericalEncoding<NetworkFloat, 2>{ workDir / "ms",
                                            config["ms_encoding"], 3 },
    });

    auto GetDim = [&config](const auto &key) {
        return config[key]["n_features_per_level"].template get<int>();
    };

    int stepNum = config["step_num"].get<int>();
    int inputDim = GetDim("volume_encoding") + GetDim("ms_encoding") +
                   GetDim("view_encoding") + GetDim("hg_encoding") +
                   GetDim("albedo_encoding") + stepNum;
    CheckError(GetDim("light_encoding") == GetDim("hg_encoding"),
               "Light encoding must have the same dimension as HG encoding");

    for (int i = 0; i < stepNum; i++)
    {
        auto root = workDir / std::format("mlp{}", i);
        networkData_->networks.emplace_back(root, config["network"], inputDim,
                                            inputDim);
    }
    networkData_->networks.emplace_back(workDir / std::format("final_mlp"),
                                        config["network_final"], inputDim, 1);
    return;
}

// Return: [lightDir, lightColor]
static std::pair<glm::vec3, glm::vec3> SelectLight(const RenderParameters &p)
{
    thread_local std::uniform_real_distribution<float> distribution;
    thread_local std::default_random_engine engine{ 244 + 24 };

    glm::vec3 lightDir = p.lightDir, lightColor = p.lightColor;
    if (p.skybox.Valid() && p.enableSkybox)
    {
        const float luminanceCoeffs[]{ 0.2126f, 0.7152f, 0.0722f };

        auto skyboxPower = p.skybox.Power();
        float directionalLightPower = 0.0f;
        for (int i = 0; i < 3; i++)
            directionalLightPower += luminanceCoeffs[i] * p.lightColor[i];
        if (skyboxPower == 0.0f && directionalLightPower == 0.0f)
            return { lightDir, lightColor };

        float probToSampleSkybox =
            skyboxPower / (directionalLightPower + skyboxPower);
        if (auto rnd = distribution(engine); rnd < probToSampleSkybox)
        { // Random draw direction by importance sampling.
            auto sampleResult = p.skybox.SampleDirection(distribution(engine),
                                                         distribution(engine));
            lightDir = sampleResult.dir;
            lightColor =
                sampleResult.radiance / (sampleResult.pdf * probToSampleSkybox);
        }
        else
        {
            lightColor /= (1 - probToSampleSkybox);
        }
    }

    return { lightDir, lightColor };
}

static RenderParametersPayload ToRenderPayload(
    const RenderParameters &p, const RenderInterface::AuxData &auxData,
    const glm::vec3 &lightDir, const glm::vec3 &lightColor)
{
    thread_local std::uniform_int_distribution<int> distribution;
    thread_local std::default_random_engine engine{
        442 + 42 // std::random_device{}()
    };

    auto forward = glm::normalize(-p.cameraPos);
    glm::vec3 right = glm::cross(glm::vec3{ 0, 1, 0 }, forward);

    auto rightLen = glm::length(right);
    if (rightLen < 1e-5) // cameraPos is ~ { 0, +1/-1, 0 }
        right = glm::vec3{ 1, 0, 0 };
    else
        right /= rightLen;
    glm::vec3 up = glm::cross(forward, right);

    float lerpCoeff = 0.0f;
    {
        using enum RenderParameters::Denoise;
        if (p.denoise == None)
            lerpCoeff = 1.0f;
        else if (p.denoise == Unbiased)
            lerpCoeff = 1.0f / (1 + p.frameNum);
        else if (p.denoise == VisualPlausible)
            lerpCoeff = std::min(0.2f, 1.0f / (1 + p.frameNum));
        else
            assert(false);
    }

    float illumScale = 0.0f;
    {
        using enum RenderParameters::Compatibility;
        if (p.compatibility == Normal)
            illumScale = 1.0f;
        else if (p.compatibility == MRPNN)
            illumScale = std::numbers::pi_v<float>;
        else if (p.compatibility == NoDirectIllum)
            illumScale = 0.0f;
        else
            assert(false);
    }

    return RenderParametersPayload{
        .frameBuffer = { .size = p.frameBuffer.size,
                         .bufferPtr = p.frameBuffer.bufferPtr,
                         .channelNum = p.frameBuffer.channelNum },
        .randomOffset = distribution(engine),
        .g = p.g,
        .albedo = p.albedo,
        .skybox = p.skybox,
        .camera = { .position = p.cameraPos,
                    .forward = forward,
                    .right = right,
                    .up = up },
        .light = { .dir = lightDir,
                   .invDir = 1.0f / lightDir,
                   .dirIsPos = glm::ivec3{ glm::greaterThanEqual(
                       lightDir, glm::vec3{ 0 }) },
                   .color = lightColor },
        .lastFrameInfo = { .camera = auxData.lastCamera,
                           .frameBuffer = { .size = { auxData.resx,
                                                      auxData.resy },
                                            .bufferPtr =
                                                auxData.lastBuffer.Get(),
                                            .channelNum = auxData.channelNum,
                                          },
                           .swapBuffer = auxData.newBuffer.Get(),
                           .lerpCoeff = lerpCoeff,
                         },
        .toneMapping = static_cast<RenderParametersPayload::ToneMapping>(
            static_cast<int>(p.toneMapping)
        ),
        .illumScale = illumScale,
        .excludeLightEncoding = p.excludeLightEncoding,
        .fastDirectIllum = p.fastDirectIllum,
        .enableSkyboxBaking = p.enableSkyboxBaking,
    };
}

extern void DispatchRenderKernel(const VolumePayload &volumePayload,
                                 const NetworkGlobalPayload &networkPayload,
                                 const RenderParametersPayload &paramPayload);

template<typename T, typename Func>
void RenderInterface::RenderBase_(const RenderParameters &params,
                                  Func renderFunc)
{
    if (params.recreateDensityTexture)
    {
        using enum RenderParameters::DensityLinearity;
        switch (params.densityInterpolation)
        {
        case Point:
            volume_.RecreateTexture(false, false);
            break;
        case Linear:
            volume_.RecreateTexture(true, false);
            break;
        case Point_MIPMapLinear:
            volume_.RecreateTexture(false, true);
            break;
        case Linear_MIPMapLinear:
            volume_.RecreateTexture(true, true);
            break;
        default:
            throw std::runtime_error{ "Unknown density interpolation." };
        }
    }

    auto [lightDir, lightColor] = SelectLight(params);
    transmittanceMap_.Update(*volume_.Get(), lightDir, params.trMapSampleLevel,
                             params.trMapMarchingSteps, params.refreshTrMap);
    auto updateGuard = auxData_->SwapBuffers(params.frameBuffer.size.x,
                                             params.frameBuffer.size.y,
                                             params.frameBuffer.channelNum);
    auxData_->lastCamera =
        renderFunc(static_cast<T &>(*this), params, lightDir, lightColor);
}

void RenderInterface::Render(const RenderParameters &params)
{
    RenderBase_<RenderInterface>(
        params, [](auto &self, const RenderParameters &params,
                   const glm::vec3 &lightDir, const glm::vec3 &lightColor) {
            auto renderPayload =
                ToRenderPayload(params, *self.auxData_, lightDir, lightColor);
            DispatchRenderKernel(VolumePayload{ *self.volume_.Get(),
                                                *self.transmittanceMap_.Get() },
                                 self.networkData_->GetPayload(),
                                 renderPayload);
            return renderPayload.camera;
        });
}

// ---------------------- RenderInterfaceWithTCNN ----------------------
struct RenderInterfaceWithTCNN::TCNNData
{
    UniqueResource<void *, Deleter<&DeleteTCNNHandler>> tcnnHandle;

    UniqueResource<NetworkFloat *, CUDADeleter> featureBuffer;
    UniqueResource<half *, CUDADeleter> radianceBuffer;
    std::size_t lastSize = 0;

    int stepNum;
    int featureDim;

    struct GBuffer
    {
        UniqueResource<glm::vec4 *, CUDADeleter> positionBuffer;
        UniqueResource<glm::vec3 *, CUDADeleter> viewDirBuffer;

        void Resize(std::size_t size)
        {
            glm::vec4 *buffer;
            CheckCUDAError(cudaMalloc(&buffer, size * sizeof(glm::vec4)));
            positionBuffer.Reset(buffer);

            glm::vec3 *buffer2;
            CheckCUDAError(cudaMalloc(&buffer2, size * sizeof(glm::vec3)));
            viewDirBuffer.Reset(buffer2);
        }

        GBufferPayload GetPayload()
        {
            return GBufferPayload{ .position = positionBuffer.Get(),
                                   .viewDir = viewDirBuffer.Get() };
        }
    } gBuffer;

    void UpdateTCNNHandle(const char *path, int albedoDim, int offsetDim)
    {
        tcnnHandle.Reset(
            NewTCNNHandler(path, featureDim, offsetDim, albedoDim));
    }

    // This is actually not strictly exception-safe, as when exception throws,
    // maybe not all buffers remain same size as fbSize.
    void TryResize(glm::u32vec2 fbSize)
    {
        auto size = (std::size_t)fbSize.x * fbSize.y;

        // Padding to ensure uniform first.
        std::size_t paddedSize =
            RoundUpDiv(size, std::size_t{ WARP_SIZE }) * WARP_SIZE;
        if (lastSize == paddedSize)
            return;

        lastSize = paddedSize;
        NetworkFloat *buffer;
        CheckCUDAError(cudaMalloc(
            &buffer, featureDim * stepNum * sizeof(NetworkFloat) * paddedSize));
        featureBuffer.Reset(buffer);

        half *buffer2;
        // 3 means channel number.
        CheckCUDAError(cudaMalloc(&buffer2, 3 * sizeof(half) * paddedSize));
        radianceBuffer.Reset(buffer2);

        // Position buffer is also used by JIT kernel, so we use paddedSize.
        // Anyway, the redudant memory is neglectible, so just pad all buffers.
        gBuffer.Resize(paddedSize);
    }

    TCNNPayload GetPayload() const noexcept
    {
        return TCNNPayload{ .handle = tcnnHandle.Get(),
                            .featureBuffer = featureBuffer.Get(),
                            .radianceBuffer = radianceBuffer.Get(),
                            .size = lastSize,
                            .featureDim = featureDim };
    }
};

RenderInterfaceWithTCNN::RenderInterfaceWithTCNN(
    const std::filesystem::path &workDir)
    : RenderInterface{ workDir }, tcnnData_{ new TCNNData{} }
{
    tcnnData_->stepNum = networkData_->networks.size() - 1;
    int offsetDim = networkData_->baseEncoding.GetFeatureDim() +
                    networkData_->shEncoding.GetFeatureDim() +
                    networkData_->viewEncoding.GetFeatureDim() +
                    networkData_->hgEncoding.GetFeatureDim();
    int albedoDim = networkData_->albedoEncoding.GetFeatureDim();
    tcnnData_->featureDim = offsetDim + albedoDim * 3 + tcnnData_->stepNum;

    tcnnData_->UpdateTCNNHandle(workDir.string().c_str(), albedoDim, offsetDim);
}

RenderInterfaceWithTCNN::RenderInterfaceWithTCNN(
    RenderInterfaceWithTCNN &&) noexcept = default;
RenderInterfaceWithTCNN &RenderInterfaceWithTCNN::operator=(
    RenderInterfaceWithTCNN &&) noexcept = default;
RenderInterfaceWithTCNN::~RenderInterfaceWithTCNN() = default;

extern void DispatchTCNNRenderKernel(
    const VolumePayload &volumePayload,
    const NetworkGlobalPayload &networkPayload,
    const RenderParametersPayload &paramPayload, const TCNNPayload &tcnnPayload,
    const GBufferPayload &gBufferPayload, EnvironmentBaking &envBaking);

void RenderInterfaceWithTCNN::Render(const RenderParameters &params)
{
    RenderBase_<RenderInterfaceWithTCNN>(
        params, [](auto &self, const RenderParameters &params,
                   const glm::vec3 &lightDir, const glm::vec3 &lightColor) {
            self.tcnnData_->TryResize(params.frameBuffer.size);

            auto renderPayload =
                ToRenderPayload(params, *self.auxData_, lightDir, lightColor);
            DispatchTCNNRenderKernel(
                VolumePayload{ *self.volume_.Get(),
                               *self.transmittanceMap_.Get() },
                self.networkData_->GetPayload(), renderPayload,
                self.tcnnData_->GetPayload(),
                self.tcnnData_->gBuffer.GetPayload(), self.auxData_->envBaking);
            return renderPayload.camera;
        });
}

void RenderInterfaceWithTCNN::EquipEnvBaking(
    const std::filesystem::path *workDir)
{
    auto &baking = auxData_->envBaking;
    if (!workDir)
    { // Clear baking.
        baking = EnvironmentBaking{};
        assert(!baking.Valid());
        return;
    }

    if (workDir->is_relative())
    {
        baking = EnvironmentBaking{ std::filesystem::path{ RELA_PATH_ROOT } /
                                    *workDir };
    }
    else
    {
        baking = EnvironmentBaking{ *workDir };
    }
    return;
}