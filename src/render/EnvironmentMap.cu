/*
* For licensing see accompanying LICENSE file.
* Copyright (C) 2026 Xiaomi Corp. All Rights Reserved.
*/

#include "EnvironmentMap.hpp"

namespace
{

static constexpr float pi = std::numbers::pi_v<float>;

__host__ __device__ glm::vec3 UVToDirection(glm::vec2 uv)
{
    float u = uv.x, v = uv.y;
    // Convert to spherical coordinates:
    //   phi (azimuth) from u: matches atan2(x, -z) mapping
    //   theta (polar) from v: matches acos(y) mapping
    float phi = u * (2.0f * pi);
    float theta = v * pi;

    // Compute direction components with proper coordinate system:
    //   x = sin(theta) * sin(phi)   [matches atan2(x, -z)]
    //   y = cos(theta)              [matches acos(y)]
    //   z = -sin(theta) * cos(phi)  [negative z for right-handed system]
    float sin_theta = sin(theta);
    return glm::vec3{ sin_theta * sin(phi), cos(theta), -sin_theta * cos(phi) };
}

} // namespace

EnvironmentMap::EnvironmentMap(const std::filesystem::path &hdriPath,
                               float exposure)
    : exposure_{ exposure }
{
    auto path =
        hdriPath.is_relative()
            ? (std::filesystem::path{ RELA_PATH_ROOT } / hdriPath).string()
            : hdriPath.string();
    int width, height, channelNum;
    if (auto buffer = stbi_loadf(path.c_str(), &width, &height, &channelNum, 0))
        image_.Reloc(buffer);
    else
        throw std::runtime_error{ "Error: fail to read image " + path };

    channelNum_ = channelNum;
    sampler_ =
        PiecewiseConstant2D{ Image{ .data = image_.Get(),
                                    .width = static_cast<unsigned int>(width),
                                    .height = static_cast<unsigned int>(height),
                                    .channelNum =
                                        static_cast<unsigned int>(channelNum) },
                             &totalIllum_ };
    // Calculate power by dividing area; this is not strictly power on sphere
    // but generally acceptable to do importance sampling.
    totalIllum_ /= (float)width * height;
    tex_ = Texture2D{ image_.Get(), width, height, channelNum };
}

auto EnvironmentMap::SampleDirection(float u1, float u2) const -> SampleResult
{
    auto result = sampler_.Sample(u1, u2);
    auto size = GetWidthAndHeight();
    auto dirCoord = glm::vec2{ result.xy.y, result.xy.x } / glm::vec2{ size };
    return { .radiance = SampleImage_(result.xy) * exposure_,
             .dir = UVToDirection(dirCoord),
             // Convert to PDF on spherical coordinates.
             .pdf = result.pdf * pi * 0.25f };
}

glm::vec3 EnvironmentMap::SampleImage_(glm::vec2 xy) const
{
    auto size = GetWidthAndHeight();
    size = { size.y, size.x }; // To make it (Height, Width) to match uv.

    auto SampleSingle = [&](std::uint32_t x, std::uint32_t y) {
        assert(x <= size.x + 1 && y <= size.y + 1);
        if (x >= size.x)
            x -= size.x;
        if (y >= size.y)
            y -= size.y;

        auto idx = std::size_t{ x } * size.y + y;
        auto ptr = image_.Get();
        if (channelNum_ == 1)
            return glm::vec3{ ptr[idx] };

        auto baseIdx = idx * channelNum_;
        return glm::vec3{ ptr[baseIdx], ptr[baseIdx + 1], ptr[baseIdx + 2] };
    };

    glm::uvec2 idx = glm::floor(xy);
    glm::vec2 offset = xy - glm::vec2{ idx };

    return glm::mix(glm::mix(SampleSingle(idx.x, idx.y),
                             SampleSingle(idx.x + 1, idx.y), offset.x),
                    glm::mix(SampleSingle(idx.x, idx.y + 1),
                             SampleSingle(idx.x + 1, idx.y + 1), offset.x),
                    offset.y);
}

EnvironmentMap::Proxy::Proxy(const EnvironmentMap &envMap)
    : exposure_{ envMap.exposure_ }, tex_{ envMap.tex_ }
{
    auto size = envMap.GetWidthAndHeight();
    deltaToMatchSampler_ = glm::vec2{ 0.5f } / glm::vec2{ size };
}
