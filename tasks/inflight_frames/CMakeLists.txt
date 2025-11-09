add_executable(inflight_frames
  main.cpp
  App.cpp
)

target_link_libraries(inflight_frames
  PRIVATE glfw etna glm::glm wsi gui render_utils scene)

target_add_shaders(inflight_frames
  shaders/toy.frag
  shaders/toy_buffer.frag
  shaders/toy.vert
)