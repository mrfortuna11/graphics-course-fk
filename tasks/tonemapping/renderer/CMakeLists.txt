
add_executable(tonemapping_renderer
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
)

target_link_libraries(tonemapping_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(tonemapping_renderer
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/perlin.comp
  shaders/quad.vert
  shaders/terrain.tesc
  shaders/terrain.tese
  shaders/normal.comp
  shaders/terrain.frag
)
