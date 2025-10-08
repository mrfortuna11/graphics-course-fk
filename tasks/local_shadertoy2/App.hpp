#pragma once

#include "shaders/UniformParams.h"

#include <etna/BlockingTransferHelper.hpp>
#include <etna/Buffer.hpp>
#include <etna/DescriptorSet.hpp>
#include <etna/GraphicsPipeline.hpp>
#include <etna/Image.hpp>
#include <etna/PerFrameCmdMgr.hpp>
#include <etna/Sampler.hpp>
#include <etna/Window.hpp>
#include <wsi/OsWindowingManager.hpp>

class App {
public:
  App();
  ~App();

  void run();

private:
  void drawFrame();

private:
  OsWindowingManager windowing;
  std::unique_ptr<OsWindow> osWindow;

  glm::uvec2 resolution;
  bool useVsync;

  std::unique_ptr<etna::Window> vkWindow;
  std::unique_ptr<etna::PerFrameCmdMgr> commandManager;
  std::unique_ptr<etna::OneShotCmdMgr> oneShotCommands;
  std::unique_ptr<etna::BlockingTransferHelper> transferHelper;

  etna::GraphicsPipeline shadertoyPipeline;
  etna::GraphicsPipeline proceduralPipeline;

  etna::Image mainImage, proceduralImage;
  etna::Image sourceTexture;
  etna::Sampler defaultSampler, detailSampler;
  etna::Buffer uniformParams;

  UniformParams params{};
};
