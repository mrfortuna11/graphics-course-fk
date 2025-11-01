add_library(common_code INTERFACE)

target_include_directories(common_code INTERFACE .)
target_shader_include_directories(common_code INTERFACE .)

target_link_libraries(common_code INTERFACE glm::glm)