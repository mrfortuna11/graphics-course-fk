
add_executable(model_bakery_renderer
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
)

target_link_libraries(model_bakery_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(model_bakery_renderer
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/postprocess.vert
  shaders/postprocess.frag
  shaders/clear_stats.comp
  shaders/minmax.comp
  shaders/histogram.comp
  shaders/reduce.comp
  shaders/skybox.vert
  shaders/skybox.frag
)
