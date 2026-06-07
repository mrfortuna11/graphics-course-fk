
add_executable(bindless_and_pbr_renderer
  main.cpp
  App.cpp
  Renderer.cpp
  WorldRenderer.cpp
  ClipmapMesh.cpp
)

target_link_libraries(bindless_and_pbr_renderer
  PRIVATE glfw etna glm::glm wsi gui scene render_utils)

target_add_shaders(bindless_and_pbr_renderer
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
  shaders/cull_count.comp
  shaders/prefix_sum.comp
  shaders/cull_write.comp
  shaders/perlin.comp
  shaders/normal.comp
  shaders/quad.vert
  shaders/terrain.tesc
  shaders/terrain.tese
  shaders/terrain.frag
  shaders/clipmap_terrain.vert
  shaders/clipmap_terrain.frag
  shaders/clipmap_terrain_depth.vert
  shaders/clipmap_terrain_depth.frag
  shaders/clipmap_fill.comp
  shaders/clipmap_splat.comp
  shaders/detail_gen.comp
  shaders/fog.comp
)
