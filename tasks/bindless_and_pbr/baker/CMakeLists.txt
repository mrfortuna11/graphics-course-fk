
add_executable(model_bakery_baker
  main.cpp
)

target_link_libraries(model_bakery_baker
  PRIVATE tinygltf
  PRIVATE glm::glm
  PRIVATE render_utils
  PRIVATE spdlog)