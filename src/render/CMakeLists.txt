add_library(RenderLib STATIC Network.cpp RenderInterface.cpp MIPMap.cu ToneMapping.cu Volume.cu
                             RenderInterface.cu Texture2D.cu PiecewiseConstant2D.cu EnvironmentMap.cu
                             EnvironmentBaking.cpp EnvironmentBaking.cu)

set_target_properties(RenderLib PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON
                      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
target_include_directories(RenderLib PRIVATE ${CMAKE_SOURCE_DIR}/external)

target_compile_definitions(RenderLib PUBLIC $<$<CONFIG:Debug>:DEBUG>)
target_compile_definitions(RenderLib PRIVATE RELA_PATH_ROOT="${CMAKE_SOURCE_DIR}")
target_compile_options(RenderLib PUBLIC "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>")

target_link_libraries(RenderLib PUBLIC nlohmann_json::nlohmann_json Utils)
target_link_libraries(RenderLib PRIVATE ExternalTCNN)