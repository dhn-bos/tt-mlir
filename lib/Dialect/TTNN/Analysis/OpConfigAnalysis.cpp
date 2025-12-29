// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAnalysis.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfigAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"
namespace mlir::tt::ttnn {

bool OpConfigAnalysis::applyOverrides() {

  // Placeholder, no overrides for now.
  //
  return false;
}
void OpConfigAnalysis::analysisImplementation() {
  // Future entrypoint for picking optimal op config.
  // Placeholder: pick the first legal config.
  //
  for (auto opConfigs : analysisInput.legalConfigs) {
    auto operation = opConfigs.first;
    std::vector<OpConfig::OpSpecificAttrs> validSpecConfigs;
    if (mlir::isa<mlir::tt::ttnn::Conv2dOp>(operation)) {
      std::vector<OpConfig> configs = opConfigs.second;
      if (opConfigs.second.size() > 1) {
        configs = [](const auto &cfgs) {
          std::vector<OpConfig> cfgsVec;
          for (const auto &cfg : cfgs) {
            if (cfg.outputLayout &&
                cfg.outputLayout.hasDRAMBufferType() &&
                cfg.outputLayout.hasInterleavedDRAMTensorMemoryLayout()) {
              cfgsVec.push_back(cfg);
            }
          }
          return cfgsVec;
        }(opConfigs.second);
        analysisResult.opLayouts[operation].push_back(configs.front().outputLayout);
        for (const auto &cfg : configs) {
          validSpecConfigs.push_back(cfg.opSpecificAttrs);
        }
        analysisResult.opSpecConfigs[operation] = validSpecConfigs;
      }
      else {
        // As a last resort pick the first legal config from the original set.
        analysisResult.opLayouts[operation].push_back(opConfigs.second.front().outputLayout);
        analysisResult.opSpecConfigs[operation].push_back(opConfigs.second.front().opSpecificAttrs);
      }
    }
    else {
    analysisResult.opLayouts[opConfigs.first].push_back(opConfigs.second[0].outputLayout);
    analysisResult.opSpecConfigs[opConfigs.first].push_back(
        opConfigs.second[0].opSpecificAttrs);
    }
  }
}
} // namespace mlir::tt::ttnn
