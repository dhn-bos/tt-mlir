// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"
#include "ttmlir/Dialect/StableHLO/Utils/GSPMDUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardyUtils.h"
#include "ttmlir/Dialect/TTCore/Utils/PopulateArgumentTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"

#include "llvm/Support/Error.h"

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_AGGRESSIVEPROPAGATIONWRAPPERPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

class AggressivePropagationWrapperPass
    : public impl::AggressivePropagationWrapperPassBase<
          AggressivePropagationWrapperPass> {
public:
  using impl::AggressivePropagationWrapperPassBase<
      AggressivePropagationWrapperPass>::AggressivePropagationWrapperPassBase;

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *context = module.getContext();

    llvm::DenseMap<mlir::BlockArgument,
                   std::pair<mlir::tt::ttcore::ShardStatus, mlir::Attribute>>
        beforeStatus;

    module.walk([&](mlir::func::FuncOp funcOp) {
      for (auto arg : funcOp.getArguments()) {
        // 1. Read ttcore.shard_status
        mlir::tt::ttcore::ShardStatus status =
            mlir::tt::ttcore::ShardStatus::Unsharded;
        if (auto attr =
                funcOp.getArgAttrOfType<mlir::tt::ttcore::ShardStatusAttr>(
                    arg.getArgNumber(), "ttcore.shard_status")) {
          status = attr.getValue();
        }

        // 2. Read sdy.sharding (nullptr if not present)
        mlir::Attribute sdyAttr =
            funcOp.getArgAttr(arg.getArgNumber(), "sdy.sharding");

        beforeStatus[arg] = {status, sdyAttr};
      }
    });

    // Propagate tensor shardings through the entire graph.
    // This propagation is taken from
    // https://github.com/openxla/shardy/blob/0b8873d121008abc3edf7db2281f2b48cc647978/docs/sdy_propagation_passes.md?plain=1#L27.
    // Aggressive propagation is a wrapper ontop of basic propagation with
    // additional options user can set. With basic propagation, only shardings
    // that have no conflicts are propagated. With aggressive propagation, we
    // can set options to resolve conflicts and propagate more shardings.
    // However, sometimes, the propagation algorithm can be too aggressive and
    // propagate shardings that are not valid. To mitigate this, we set
    // conservativePropagation to true, which ensures that only shardings that
    // are valid are propagated.
    mlir::sdy::PropagationOptions propagationOptions;
    mlir::sdy::PropagationStrategy propagationStrategy =
        mlir::sdy::PropagationStrategy::Aggressive;
    propagationOptions.conservativePropagation = true;
    mlir::PassManager pm(context);
    pm.addPass(mlir::sdy::createAggressivePropagationPass(propagationOptions,
                                                          propagationStrategy));
    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }
    
    module.walk([&](mlir::func::FuncOp funcOp) {
      for (auto arg : funcOp.getArguments()) {
        auto it = beforeStatus.find(arg);
        if (it == beforeStatus.end()) {
          continue;
        }

        mlir::Attribute beforeSdy = it->second.second;
        mlir::Attribute afterSdy =
            funcOp.getArgAttr(arg.getArgNumber(), "sdy.sharding");

        bool changed = (beforeSdy && afterSdy) ? beforeSdy != afterSdy
                                               : (beforeSdy || afterSdy);

        if (changed) {
          funcOp.setArgAttr(
              arg.getArgNumber(), "ttcore.shard_status",
              mlir::tt::ttcore::ShardStatusAttr::get(
                  context, mlir::tt::ttcore::ShardStatus::Unsharded));
        }
      }
    });

    return;
  }
};
} // namespace mlir::tt::stablehlo
