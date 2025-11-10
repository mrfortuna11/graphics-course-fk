add_executable(terrain_renderer
    main.cpp    
    App.cpp
    Renderer.cpp
    WorldRenderer.cpp
)

target_link_libraries(terrain_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(terrain_renderer
  shaders/static_mesh.frag
  shaders/static_mesh.vert
  shaders/perlin.comp
  shaders/terrain.tesc
  shaders/terrain.tese
)