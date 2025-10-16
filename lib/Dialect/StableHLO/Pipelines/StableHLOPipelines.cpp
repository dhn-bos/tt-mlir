// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Pipelines/StableHLOPipelines.h"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"

#include "mlir/Transforms/Passes.h"

namespace mlir::tt::stablehlo {
//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void createStableHLOPipeline(OpPassManager &pm,
                             const StableHLOPipelineOptions &options) {
  // Inline all operations to make analysis easier.
  pm.addPass(mlir::createInlinerPass());

  // Annotate arguments with tt tensor annotations if the exist.
  pm.addPass(
      mlir::tt::ttcore::createTTPopulateArgumentTypes(options.argumentTypeMap));

  // Annotate arguments with whether they are already pre-sharded or not.
  pm.addPass(createApplyArgumentShardStatusPass());

  // Convert any xla.sdy ops to sdy ops.
  pm.addPass(createConvertXlaSdyToSdyPass());

  // Analyze the mesh of the graph and update shardings or annotations to match
  // the target device.
  AnalyzeMeshPassOptions analyzeMeshOptions;
  analyzeMeshOptions.meshShape = llvm::to_vector(options.meshShape);
  analyzeMeshOptions.automaticArgAnalysis = options.automaticArgAnalysis;
  pm.addPass(createAnalyzeMeshPass(analyzeMeshOptions));

  // Apply sharding constraints conditionally.
  pm.nest<mlir::func::FuncOp>().addPass(createApplyShardingConstraintsPass());

  // Propagate tensor shardings through the entire graph conditionally.
  // This propagation is taken from
  // https://github.com/openxla/shardy/blob/0b8873d121008abc3edf7db2281f2b48cc647978/docs/sdy_propagation_passes.md?plain=1#L27.
  // Aggressive propagation is a wrapper ontop of basic propagation with
  // additional options user can set. With basic propagation, only shardings
  // that have no conflicts are propagated. With aggressive propagation, we can
  // set options to resolve conflicts and propagate more shardings. However,
  // sometimes, the propagation algorithm can be too aggressive and propagate
  // shardings that are not valid. To mitigate this, we set
  // conservativePropagation to true, which ensures that only shardings that are
  // valid are propagated.
  pm.addPass(createAggressivePropagationPass());

  // Convert sharding constraints to reshards conditionally.
  pm.nest<mlir::func::FuncOp>().addPass(
      createShardingConstraintToReshardPass());

  // Insert explicit reshards conditionally.
  pm.nest<mlir::func::FuncOp>().addPass(createInsertExplicitReshardsPass());

  // Wrap all operations under a sdy manual computation op to allow conversion
  // from stablehlo into ttir.
  pm.addPass(createWrapUnderManualComputationPass());

  // Convert reshards to collectives conditionally.
  pm.nest<mlir::func::FuncOp>().addPass(createReshardToCollectivesPass());

  // Split tensor dimensions according to tensor sharding annotations.
  pm.addPass(createUpdateGlobalToLocalShapesPass());

  // Close tensor shardings as analysis is complete conditionally.
  pm.addPass(createCloseShardingsPass());

  // Run canonicalizer pass.
  pm.addPass(mlir::createCanonicalizerPass());
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerStableHLOPipeline() {
  // StableHLO Pipeline
  mlir::PassPipelineRegistration<mlir::tt::stablehlo::StableHLOPipelineOptions>(
      "stablehlo-pipeline",
      "StableHLO pipeline to run stablehlo and shardy specific passes",
      mlir::tt::stablehlo::createStableHLOPipeline);
}

} // namespace mlir::tt::stablehlo
