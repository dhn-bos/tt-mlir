// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/Analysis/DestRegisterAnalysis.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MINSERTDSTREGISTERACCESS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
struct D2MInsertDstRegisterAccessRewriter final
    : public OpRewritePattern<GenericOp> {
public:
  D2MInsertDstRegisterAccessRewriter(mlir::MLIRContext *ctx, bool useTileMatmul,
                                     unsigned maxDstPhysicalSizeTiles,
                                     const DestRegisterAnalysis *analysis)
      : OpRewritePattern<GenericOp>(ctx), useTileMatmul(useTileMatmul),
        maxDstPhysicalSizeTiles(maxDstPhysicalSizeTiles), analysis(analysis) {};

  template <typename OpT>
  using OpAndIndexOffset = std::pair<OpT, int64_t>;

  // Stores dst loads/stores, organized by common loop nests.
  struct CopyInfo {
    void push_back(affine::AffineLoadOp load, int64_t indexOffset) {
      loads.emplace_back(load, indexOffset);
    }

    void push_back(affine::AffineStoreOp store, int64_t indexOffset) {
      stores.emplace_back(store, indexOffset);
    }

    SmallVector<int64_t> guardIndices;
    SmallVector<OpAndIndexOffset<affine::AffineLoadOp>> loads;
    SmallVector<OpAndIndexOffset<affine::AffineStoreOp>> stores;
  };
  using CopyInfoMap = DenseMap<Operation *, CopyInfo>;

  // class DstSliceAllocationState {
  // public:
  //   int64_t allocate() { return nextSliceIndex++; }

  //   void setStoreToDst() { storedToDst = true; }
  //   bool didStoreToDst() { return storedToDst; }
  //   int64_t getCurrSliceIndex() { return nextSliceIndex - 1; }

  // private:
  //   int64_t nextSliceIndex = 0;
  //   bool storedToDst = false;
  // };

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (unsigned regionIndex = 0; regionIndex < op.getNumRegions();
         regionIndex++) {
      if (op.getRegionThreadType(regionIndex) != ThreadType::Compute) {
        continue;
      }

      Region &region = op.getRegion(regionIndex);
      Block &block = region.getBlocks().front();

      Type largestDstType = utils::getRegionLargestDstElemType(region);
      const unsigned dstCapacity =
          ttcore::getOpChipDescAttr(op).getDstLogicalSizeTiles(
              largestDstType, false, maxDstPhysicalSizeTiles);

      bool linalgToAffineFailed = false;
      block.walk([&](linalg::GenericOp linalgGenericOp) {
        llvm::errs() << "DEBUG: Processing linalg::GenericOp (Inputs: "
                     << linalgGenericOp.getInputs().size()
                     << ", Outputs: " << linalgGenericOp.getOutputs().size()
                     << ")\n";
        if (!useTileMatmul && hasTileMatmul(linalgGenericOp)) {
          linalgToAffineFailed |= rewriteTileMatmulAsTileMatmulBlock(
              rewriter, op, region, linalgGenericOp, dstCapacity, modified,
              analysis, genericOpIndex);
          genericOpIndex++;
          return;
        }

        rewriter.setInsertionPoint(linalgGenericOp);
        // Apply linalg to affine loops pass.
        auto linalgLoops =
            linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
        if (failed(linalgLoops)) {
          linalgToAffineFailed = true;
          return;
        }

        // Extract maxDstUsage and dstSliceIndices from analysis before erasing
        int maxDstUsage = 0;
        SmallVector<int> dstSliceIndices;
        if (genericOpIndex < analysis->dstRegisterInfoList.size()) {
          const auto &info =
              analysis
                  ->dstRegisterInfoList[analysis->dstRegisterInfoList.size() -
                                        1 - genericOpIndex];
          maxDstUsage = info.dstMaxUsage;
          dstSliceIndices = info.dstSliceIndices;
          llvm::errs() << "DEBUG: Got info for genericOpIndex "
                       << genericOpIndex << "\n"
                       << "  maxDstUsage: " << maxDstUsage << "\n"
                       << "  dstSliceIndices: ";
          for (int idx : dstSliceIndices) {
            llvm::errs() << idx << " ";
          }
          llvm::errs() << "\n";
        } else {
          llvm::errs() << "DEBUG: genericOpIndex " << genericOpIndex
                       << " out of bounds for analysis list (size: "
                       << analysis->dstRegisterInfoList.size() << ")\n";
        }

        rewriter.eraseOp(linalgGenericOp);
        modified |= insertDstRegisterAccess(
            rewriter, op, region, dstCapacity, maxDstUsage, dstSliceIndices,
            !linalgLoops.value().empty() ? linalgLoops.value().front()
                                         : nullptr);
        genericOpIndex++;
      });
      if (linalgToAffineFailed) {
        return failure();
      }
    }
    return success(modified);
  }

  static bool
  insertDstRegisterAccess(PatternRewriter &rewriter, GenericOp op,
                          Region &region, unsigned dstCapacity, int maxDstUsage,
                          const SmallVector<int> &dstSliceIndices,
                          Operation *outermostInnerComputeLoop = nullptr) {
    assert(region.getBlocks().size() == 1);
    if (hasAcquireDstOp(region)) {
      return false;
    }

    Location loc = op.getLoc();

    // 1. Collect all loads/stores to dst organized by loop nest.
    auto [copyInfos, dstAllocation] = collectDstAccesses(
        op, region, outermostInnerComputeLoop, dstSliceIndices);
    if (copyInfos.empty()) {
      return false;
    }

    // 2. Insert acquire dst.
    AcquireDstOp acquireDst =
        insertAcquireDst(rewriter, loc, region, copyInfos,
                         outermostInnerComputeLoop, dstCapacity);
    Value dst = acquireDst.getResult();

    // 3. Generate data copy loops to/from dst and output cb.
    dataCopyGenerate(rewriter, loc, dst, copyInfos);

    // 4. Rewrite stores to use dst register based on allocation.
    // TODO: For now, just use the first entry in genericOpMap since we're
    // dealing with a single d2m.generic operation. In the future, we should
    // match the correct GenericOp from the analysis.
    assert(!dstSliceIndices.empty() && "No generic ops in analysis");
    insertDstRegisterAllocation(rewriter, loc, dst, dstAllocation);

    return true;
  }

  static bool hasAcquireDstOp(Region &region) {
    return !region.getOps<AcquireDstOp>().empty();
  }

  static std::pair<MemRefType, int64_t>
  inferCbInfoFromAllAccesses(const CopyInfoMap &copyInfos) {
    MemRefType canonicalType = nullptr;
    int64_t maxDstSliceIdx = -1;

    for (auto [loopNest, copyInfo] : copyInfos) {
      for (auto &[loadOp, idx] : copyInfo.loads) {
        if (canonicalType == nullptr) {
          canonicalType = loadOp.getMemRefType();
        } else {
          TT_assertv(loadOp.getMemRefType().getShape() ==
                         canonicalType.getShape(),
                     "Multiple interpretations of DST not supported.");
        }
        maxDstSliceIdx = std::max(maxDstSliceIdx, idx);
      }
      for (auto &[storeOp, idx] : copyInfo.stores) {
        if (canonicalType == nullptr) {
          canonicalType = storeOp.getMemRefType();
        } else {
          TT_assertv(storeOp.getMemRefType().getShape() ==
                         canonicalType.getShape(),
                     "Multiple interpretations of DST not supported.");
        }
        maxDstSliceIdx = std::max(maxDstSliceIdx, idx);
      }
    }
    TT_assert(canonicalType != nullptr);
    TT_assert(maxDstSliceIdx >= 0);
    return {canonicalType, maxDstSliceIdx};
  }

  static AcquireDstOp insertAcquireDst(PatternRewriter &rewriter, Location loc,
                                       Region &region,
                                       const CopyInfoMap &copyInfos,
                                       Operation *outermostInnerComputeLoop,
                                       unsigned dstCapacity) {
    assert(!copyInfos.empty());
    if (outermostInnerComputeLoop) {
      rewriter.setInsertionPoint(outermostInnerComputeLoop);
    } else {
      rewriter.setInsertionPointToStart(&region.front());
    }

    auto [cbType, maxDstSliceIdx] = inferCbInfoFromAllAccesses(copyInfos);
    // Calculate dst shape as N slices of cb shape.
    const int64_t volume = ttmlir::utils::volume(cbType.getShape());
    TT_assert(volume <= dstCapacity);
    const int64_t numDstSlices = dstCapacity / volume;
    // TT_assertv(maxDstSliceIdx < numDstSlices,
    //  "Insufficient DST capacity for all operands.");
    SmallVector<int64_t> dstShape({numDstSlices});
    dstShape.append(cbType.getShape().begin(), cbType.getShape().end());
    MemRefType dstType =
        MemRefType::get(dstShape, cbType.getElementType(),
                        mlir::AffineMap::getMultiDimIdentityMap(
                            dstShape.size(), rewriter.getContext()),
                        rewriter.getAttr<ttcore::MemorySpaceAttr>(
                            ttcore::MemorySpace::RegisterDst));

    return rewriter.create<AcquireDstOp>(loc, dstType);
  }

  // Walk all compute ops in the region and collect all dst accesses organized
  // by loop nest. Also maintain dst register allocation state such that
  // multiple operands get unique dst indices. Currently this routine only does
  // register allocation for loads and just assumes that stores get exclusive
  // access. Returns a map of loop nest -> copy info, which contains a list of
  // loads and stores to copy into hoisted loop nests.

  // Maps each D2MGenericRegionComputeOpTrait operation result to a dest
  // register slice index and its containing loop nest.
  struct DstRegisterInfo {
    int64_t dstSliceIndex;
    Operation *outermostLoop;
  };
  using DstRegisterAllocation = DenseMap<Operation *, DstRegisterInfo>;

  // Struct to hold the results of dst access collection.
  struct DstAccessCollection {
    CopyInfoMap copyInfos;
    DstRegisterAllocation dstAllocation;
  };

  // Return both the copy nest info and dst allocation info.
  static DstAccessCollection
  collectDstAccesses(GenericOp op, Region &region,
                     Operation *outermostInnerComputeLoop,
                     const SmallVector<int> &dstSliceIndices) {
    CopyInfoMap copyInfos;
    // DstSliceAllocationState dstSliceAllocationState;
    DstRegisterAllocation dstRegisterAllocation;

    size_t dstSliceIdxCounter = 0;
    region.walk([&](OperandLoadStoreRegisterOpInterface computeOp) {
      llvm::errs() << "DEBUG: Processing compute op #" << dstSliceIdxCounter
                   << " (Op: " << computeOp->getName().getStringRef()
                   << ", Operands: " << computeOp->getNumOperands()
                   << ", Users: "
                   << std::distance(computeOp->user_begin(),
                                    computeOp->user_end())
                   << ")\n";
      // We're generating loads and stores for dst, so we can ignore loads and
      // stores that are already on dst.
      auto notDstMemspace = [](auto op) {
        return op && ttcore::getMemorySpace(op.getMemRef()) !=
                         ttcore::MemorySpace::RegisterDst;
      };

      // Collect loads to this op.
      for (int64_t operandIdx : computeOp.getOperandsLoadFromDstRegister()) {
        if (auto potentialLoad = computeOp->getOperand(operandIdx)
                                     .getDefiningOp<affine::AffineLoadOp>();
            notDstMemspace(potentialLoad)) {
          // Use DST slice index from analysis
          if (dstSliceIdxCounter >= dstSliceIndices.size()) {
            llvm::errs() << "ERROR: dstSliceIdxCounter (" << dstSliceIdxCounter
                         << ") out of bounds for dstSliceIndices (size: "
                         << dstSliceIndices.size() << ") during load access\n";
            return;
          }
          int64_t dstSliceIndex = dstSliceIndices[dstSliceIdxCounter];
          llvm::errs() << "  Accessing dstSliceIndices[" << dstSliceIdxCounter
                       << "] = " << dstSliceIndex << "\n";
          collectDstAccess<affine::AffineLoadOp>(op, potentialLoad, copyInfos,
                                                 dstSliceIndex,
                                                 outermostInnerComputeLoop);
          dstSliceIdxCounter++;
        }
      }

      // Collect stores from this op.
      for (auto *user : computeOp->getUsers()) {
        if (auto potentialStore = mlir::dyn_cast<affine::AffineStoreOp>(user);
            notDstMemspace(potentialStore)) {

          // assert(!dstSliceAllocationState.didStoreToDst() &&
          //        "Multiple stores from last op to dst not supported");

          auto dstRegInPlace = computeOp.getDstRegInPlace();
          int64_t dstSliceIndex = -1;
          if (dstRegInPlace) {
            bool isUnaryOp = computeOp->getNumOperands() == 1;
            bool isTileMatmul = mlir::isa<d2m::TileMatmulOp>(computeOp);
            bool isReduction = mlir::isa<d2m::TileReduceMaxOp>(computeOp) ||
                               mlir::isa<d2m::TileReduceSumOp>(computeOp);
            assert((isUnaryOp || isTileMatmul || isReduction) &&
                   "Only unary ops, tile matmul, and reductions supported for "
                   "destination register in "
                   "place, multi-operand ops would reference wrong tile, but "
                   "those ops should be setting output tile.");
            // Use index from analysis for in-place ops
            if (dstSliceIndices.empty()) {
              llvm::errs() << "ERROR: dstSliceIndices is empty when accessing "
                           << "last element for in-place store\n";
              return;
            }
            dstSliceIndex = dstSliceIndices[dstSliceIndices.size() - 1];
            llvm::errs() << "  In-place store accessing dstSliceIndices["
                         << (dstSliceIndices.size() - 1)
                         << "] = " << dstSliceIndex << "\n";
            // dstSliceIdxCounter < dstSliceIndices.size()
            //                     ?
            //                     :
            //                     dstSliceAllocationState.getCurrSliceIndex();
          } else {
            // Use index from analysis
            if (dstSliceIndices.empty()) {
              llvm::errs() << "ERROR: dstSliceIndices is empty when accessing "
                           << "last element for non-in-place store\n";
              return;
            }
            dstSliceIndex = dstSliceIndices[dstSliceIndices.size() - 1];
            llvm::errs() << "  Non-in-place store accessing dstSliceIndices["
                         << (dstSliceIndices.size() - 1)
                         << "] = " << dstSliceIndex << "\n";
            // dstSliceIndex = dstSliceIdxCounter < dstSliceIndices.size()
            //                     ? dstSliceIndices[dstSliceIdxCounter]
            //                     : dstSliceAllocationState.allocate();
            // dstSliceAllocationState.setStoreToDst();
          }
          collectDstAccess<affine::AffineStoreOp>(op, potentialStore, copyInfos,
                                                  dstSliceIndex,
                                                  outermostInnerComputeLoop);

        }
        // If the user isn't a store, it must be another compute consumer and we
        // need to set or allocate a dest register intermediate for it.
        else {
          assert(user->hasTrait<D2MGenericRegionComputeOpTrait>());
          assert(computeOp->hasOneUse() &&
                 "Currently we do not support multiple "
                 "users in the same compute dst region.");
          assert(computeOp->getNumResults() == 1);
          assert(!dstRegisterAllocation.contains(computeOp));
          // If op stores to dst in place, we don't need to allocate a new dst
          // register, just use the current dst index from analysis.

          if (dstSliceIdxCounter >= dstSliceIndices.size()) {
            llvm::errs() << "ERROR: dstSliceIdxCounter (" << dstSliceIdxCounter
                         << ") out of bounds for dstSliceIndices (size: "
                         << dstSliceIndices.size()
                         << ") when allocating compute op result\n";
            return;
          }
          int32_t allocatedIndex = dstSliceIndices[dstSliceIdxCounter];
          llvm::errs() << "  Compute op result allocation using "
                       << "dstSliceIndices[" << dstSliceIdxCounter
                       << "] = " << allocatedIndex << "\n";
          // computeOp.getDstRegInPlace()
          //     ? (dstSliceIdxCounter < dstSliceIndices.size()
          //            ? dstSliceIndices[dstSliceIdxCounter]
          //            : dstSliceAllocationState.getCurrSliceIndex())
          //     : (dstSliceIdxCounter < dstSliceIndices.size()
          //            ? dstSliceIndices[dstSliceIdxCounter]
          //            : dstSliceAllocationState.allocate());
          // dstSliceIdxCounter++;

          dstRegisterAllocation[computeOp] = {allocatedIndex,
                                              outermostInnerComputeLoop};
        }
      }

      // Move to next DST slice index for next compute op
      llvm::errs() << "DEBUG: Incrementing counter from " << dstSliceIdxCounter
                   << " to " << (dstSliceIdxCounter + 1) << "\n";
      dstSliceIdxCounter++;
    });
    return {copyInfos, dstRegisterAllocation};
  }

  static BlockArgument lookThroughSubView(Value memref) {
    while (auto subView = mlir::dyn_cast_or_null<memref::SubViewOp>(
               memref.getDefiningOp())) {
      memref = subView.getSource();
    }
    return mlir::cast<BlockArgument>(memref);
  }

  // Collect a single load or store to dst organized by loop nest.
  template <typename LoadOrStoreOp>
  static void collectDstAccess(GenericOp op, LoadOrStoreOp loadOrStore,
                               CopyInfoMap &copyInfos,
                               int64_t nextDstSliceIndex,
                               Operation *outermostInnerComputeLoop) {
    if (!outermostInnerComputeLoop) {
      // If there is no outermostInnerComputeLoop, the common ancestor is the
      // operation itself.
      outermostInnerComputeLoop = loadOrStore;
    }

    auto [iter, inserted] = copyInfos.try_emplace(outermostInnerComputeLoop);
    CopyInfo &copyInfo = iter->second;
    copyInfo.push_back(loadOrStore, nextDstSliceIndex);
    SmallVector<int64_t> guardIndices = op.getNonParticipatingLoopDims(
        lookThroughSubView(loadOrStore.getMemRef()).getArgNumber());
    if (inserted) {
      // First access in this loop nest - set the guard indices.
      copyInfo.guardIndices = guardIndices;
    } else {
      // Subsequent access - verify guard indices are the same.
      assert(
          guardIndices == copyInfo.guardIndices &&
          "Expected same guard indices across all accesses in this loop nest.");
    }
  }

  static bool hasTileMatmul(linalg::GenericOp linalgGenericOp) {
    bool hasTileMatmul = false;
    linalgGenericOp->walk([&](d2m::TileMatmulOp) {
      hasTileMatmul = true;
      return WalkResult::interrupt();
    });
    return hasTileMatmul;
  }
  /*
    Expand a linalg.generic op that contains a tile_matmul into a
    tile_matmul_block.

    - Uses the linalg.generic and affine semantics to generate copy/pack loops.
    - Deletes the compute loop nest since tile_matmul_block includes the loops
    inside it.
  */
  static bool rewriteTileMatmulAsTileMatmulBlock(
      PatternRewriter &rewriter, GenericOp op, Region &region,
      linalg::GenericOp linalgGenericOp, unsigned dstCapacity, bool &modified,
      const DestRegisterAnalysis *analysis, size_t genericOpIndex) {
    assert(linalgGenericOp.getInputs().size() == 2 &&
           "Expected exactly 2 input for tile matmul");
    assert(linalgGenericOp.getOutputs().size() == 1 &&
           "Expected exactly 1 output for tile matmul");

    Value inputAMemref = linalgGenericOp.getInputs()[0];
    Value inputBMemref = linalgGenericOp.getInputs()[1];
    Value outputCMemref = linalgGenericOp.getOutputs()[0];

    rewriter.setInsertionPoint(linalgGenericOp);

    auto linalgLoops = linalg::linalgOpToAffineLoops(rewriter, linalgGenericOp);
    if (failed(linalgLoops)) {
      return false;
    }

    // Extract maxDstUsage and dstSliceIndices from analysis before erasing
    int maxDstUsage = 0;
    SmallVector<int> dstSliceIndices;
    if (analysis) {
      const auto &info =
          analysis->dstRegisterInfoList[analysis->dstRegisterInfoList.size() -
                                        1 - genericOpIndex];
      maxDstUsage = info.dstMaxUsage;
      dstSliceIndices = info.dstSliceIndices;
      llvm::errs()
          << "DEBUG (TileMatmul): Found linalgGenericOp in analysis map\n"
          << "  maxDstUsage: " << maxDstUsage << "\n"
          << "  Input count: " << linalgGenericOp.getInputs().size() << "\n"
          << "  Output count: " << linalgGenericOp.getOutputs().size() << "\n"
          << "  dstSliceIndices: ";
      for (int idx : dstSliceIndices) {
        llvm::errs() << idx << " ";
      }
      llvm::errs() << "\n";
    } else {
      llvm::errs() << "DEBUG (TileMatmul): analysis is null\n";
    }

    rewriter.eraseOp(linalgGenericOp);
    modified |= insertDstRegisterAccess(
        rewriter, op, region, dstCapacity, maxDstUsage, dstSliceIndices,
        !linalgLoops.value().empty() ? linalgLoops.value().front() : nullptr);

    Operation *outerLoop = linalgLoops.value()[0];
    Block *parentBlk = outerLoop->getBlock();
    auto insertPos = std::next(Block::iterator(outerLoop));

    rewriter.setInsertionPoint(parentBlk, insertPos);
    for (Operation *loopOp : llvm::reverse(linalgLoops.value())) {
      rewriter.eraseOp(loopOp);
    }
    rewriter.create<d2m::TileMatmulBlockOp>(op.getLoc(), inputAMemref,
                                            inputBMemref, outputCMemref);
    return true;
  }

  static void dataCopyGenerate(PatternRewriter &rewriter, Location loc,
                               Value dst, const CopyInfoMap &copyInfos) {
    for (const auto &[loopNestOrOp, copyInfo] : copyInfos) {
      // Save this insertion point as loopNestOrOp may be replaced.
      rewriter.setInsertionPointAfter(loopNestOrOp);
      auto insertionPointAfterLoopNest = rewriter.saveInsertionPoint();

      rewriter.setInsertionPoint(loopNestOrOp);
      auto guard = insertGuardForLoopNest(rewriter, loc, copyInfo.guardIndices);
      if (guard) {
        rewriter.setInsertionPointToStart(&guard.getThenRegion().front());
      }
      dataCopyGenerate<affine::AffineLoadOp>(
          rewriter, loopNestOrOp, copyInfo.loads,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto l1Load = rewriter.create<affine::AffineLoadOp>(
                loc, cb, l1AccessMap, l1AccessIndices);
            rewriter.create<affine::AffineStoreOp>(
                loc, l1Load.getResult(), dst, dstAccessMap, dstAccessIndices);
          },
          // Replacement of the original load with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineLoadOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            rewriter.replaceOpWithNewOp<affine::AffineLoadOp>(
                op, dst, dstAccessMap, dstAccessIndices);
          });

      rewriter.restoreInsertionPoint(insertionPointAfterLoopNest);
      dataCopyGenerate<affine::AffineStoreOp>(
          rewriter, loopNestOrOp, copyInfo.stores,
          // Load/store dst access generation.
          [&](PatternRewriter &rewriter, Location loc, Value cb,
              AffineMap l1AccessMap, ValueRange l1AccessIndices,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            auto dstLoad = rewriter.create<affine::AffineLoadOp>(
                loc, dst, dstAccessMap, dstAccessIndices);
            Value valueToStore = dstLoad.getResult();

            // Insert dst reinterpret cast if destination CB type differs
            // from dst type
            auto cbType = mlir::cast<MemRefType>(cb.getType());
            if (valueToStore.getType() != cbType.getElementType()) {
              valueToStore = rewriter
                                 .create<d2m::DstReinterpretCastOp>(
                                     loc, cbType.getElementType(), valueToStore)
                                 .getResult();
            }

            rewriter.create<affine::AffineStoreOp>(
                loc, valueToStore, cb, l1AccessMap, l1AccessIndices);
          },
          // Replacement of the original store with one from dst.
          [&](PatternRewriter &rewriter, affine::AffineStoreOp op,
              AffineMap dstAccessMap, ValueRange dstAccessIndices) {
            Value valueToStore = op.getValue();
            // Insert dst reinterpret cast if value type differs from dst
            // type
            auto dstType = mlir::cast<MemRefType>(dst.getType());
            if (valueToStore.getType() != dstType.getElementType()) {
              valueToStore =
                  rewriter
                      .create<d2m::DstReinterpretCastOp>(
                          op.getLoc(), dstType.getElementType(), valueToStore)
                      .getResult();
            }
            rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(
                op, valueToStore, dst, dstAccessMap, dstAccessIndices);
          });
    }
  }

  static scf::IfOp insertGuardForLoopNest(PatternRewriter &rewriter,
                                          Location loc,
                                          ArrayRef<int64_t> guardIndices) {
    if (guardIndices.empty()) {
      return nullptr;
    }
    auto zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto cmp = rewriter
                   .create<arith::ConstantOp>(loc, rewriter.getI1Type(),
                                              rewriter.getBoolAttr(false))
                   .getResult();
    for (int64_t index : guardIndices) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(loc, index);
      auto eq = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               iterIndex, zero);
      cmp = rewriter.create<arith::OrIOp>(loc, cmp, eq).getResult();
    }
    return rewriter.create<scf::IfOp>(loc, cmp);
  }

  template <typename LoadStoreOpTy>
  static void dataCopyGenerate(
      PatternRewriter &rewriter, Operation *loopNestOrOp,
      ArrayRef<OpAndIndexOffset<LoadStoreOpTy>> loadStoreOps,
      llvm::function_ref<void(PatternRewriter &, Location, Value, AffineMap,
                              ValueRange, AffineMap, ValueRange)>
          loadStoreDstAccessGenerator,
      llvm::function_ref<void(PatternRewriter &, LoadStoreOpTy, AffineMap,
                              ValueRange)>
          dstAccessReplacement) {
    if (loadStoreOps.empty()) {
      return;
    }

    mlir::IRMapping irMapper;
    // Only Clone loop nests if a loop exists.
    if (mlir::isa<affine::AffineForOp>(loopNestOrOp)) {
      rewriter.clone(*loopNestOrOp, irMapper)->walk([&](Operation *op) {
        // Erase the loop bodies except for other nested loops / yields.
        if (!mlir::isa<affine::AffineForOp, affine::AffineYieldOp,
                       affine::AffineApplyOp>(op)) {
          op->dropAllUses();
          rewriter.eraseOp(op);
        }
      });
    }

    for (auto [loadStore, dstSliceIndex] : loadStoreOps) {
      Block *fromScope = loadStore->getBlock();
      Block *toScope = irMapper.lookupOrNull(fromScope);
      if (toScope) {
        Operation *terminator = toScope->getTerminator();
        if (terminator) {
          rewriter.setInsertionPoint(terminator);
        } else {
          rewriter.setInsertionPointToEnd(toScope);
        }
      }

      // Generate the data copy loop for the load store.
      {
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), irMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap());
        loadStoreDstAccessGenerator(
            rewriter, loadStore.getLoc(), loadStore.getMemRef(), l1AccessMap,
            l1AccessIndices, dstAccessMap, dstAccessIndices);
      }

      // Replace the original load store with one from dst.
      {
        // Empty IR mapper because we want to preserve original loop vars.
        mlir::IRMapping dummyIRMapper;
        rewriter.setInsertionPoint(loadStore);
        auto [l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices] =
            buildIndices(rewriter, loadStore.getLoc(), dummyIRMapper,
                         loadStore.getIndices(), dstSliceIndex,
                         loadStore.getMap());
        dstAccessReplacement(rewriter, loadStore, dstAccessMap,
                             dstAccessIndices);
      }
    }
  }

  // Extract loop induction variables from the outermost loop operation.
  // This collects induction variables from all nested loops in the nest.
  static SmallVector<Value> extractLoopInductionVars(Operation *outermostLoop) {
    SmallVector<Value> loopInductionVars;
    if (!outermostLoop) {
      return loopInductionVars;
    }

    // Collect induction variables from all loops in the nest.
    outermostLoop->walk([&](affine::AffineForOp loop) {
      loopInductionVars.push_back(loop.getBody()->getArgument(0));
    });

    // Reverse to get innermost loops first.
    std::reverse(loopInductionVars.begin(), loopInductionVars.end());
    return loopInductionVars;
  }

  // Rewrite stores to use dst register based on allocation map.
  static void insertDstRegisterAllocation(
      PatternRewriter &rewriter, Location loc, Value dst,
      const DstRegisterAllocation &dstRegisterAllocation) {
    auto dstType = dyn_cast<MemRefType>(dst.getType());
    if (!dstType) {
      return;
    }
    const unsigned dstRank = dstType.getRank();

    // Iterate directly through dst register allocation entries.
    for (const auto &[op, dstInfo] : dstRegisterAllocation) {
      int64_t dstSliceIndex = dstInfo.dstSliceIndex;
      SmallVector<Value> loopInductionVars =
          extractLoopInductionVars(dstInfo.outermostLoop);

      // Store the result of this operation to dst register.
      rewriter.setInsertionPoint(op);

      SmallVector<Value> storeIndices;

      // Build store indices: [dstSliceIndex, loop_vars..., 0, 0, ...] using
      // loop induction variables for the dimensions that correspond to loops.
      storeIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, dstSliceIndex));

      // Use induction variables from the allocation.
      storeIndices.append(loopInductionVars);

      // Pad with zeros for remaining dimensions.
      while (storeIndices.size() < dstRank) {
        storeIndices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }

      // Ensure storeIndices matches the destination memref rank.
      assert(storeIndices.size() == dstRank &&
             "storeIndices size must match destination memref rank. If it's "
             "greater, probably need to use getNonParticipatingLoopDims to "
             "skip loop dimensions: "
             "https://github.com/tenstorrent/tt-mlir/pull/"
             "5081#discussion_r2376709558");

      auto storeMap =
          AffineMap::getMultiDimIdentityMap(dstRank, rewriter.getContext());

      rewriter.setInsertionPointAfter(op);

      // Insert dst reinterpret cast if compute result type differs from
      // dst type
      Value originalResult = op->getResult(0);
      Type originalType = originalResult.getType();
      Value valueToStore = originalResult;
      Operation *castOp = nullptr;
      bool needsTypeCast = (originalType != dstType.getElementType());

      if (needsTypeCast) {
        auto cast = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, dstType.getElementType(), valueToStore);
        valueToStore = cast.getResult();
        castOp = cast.getOperation();
      }

      auto storeOp = rewriter.create<affine::AffineStoreOp>(
          loc, valueToStore, dst, storeMap, storeIndices);

      auto loadedResult = rewriter.create<affine::AffineLoadOp>(
          loc, dst, storeMap, storeIndices);

      // If we cast for storage, we need to cast back to the original type
      // after loading, since downstream ops expect the original type.
      Value replacementValue = loadedResult.getResult();
      Operation *castBackOp = nullptr;
      if (needsTypeCast) {
        auto castBack = rewriter.create<d2m::DstReinterpretCastOp>(
            loc, originalType, replacementValue);
        replacementValue = castBack.getResult();
        castBackOp = castBack.getOperation();
      }

      // Replace all uses of the original result with the (possibly cast back)
      // loaded result from dst register, but exclude the store operation and
      // cast operations to avoid circular dependencies.
      rewriter.replaceUsesWithIf(
          originalResult, replacementValue, [&](mlir::OpOperand &operand) {
            Operation *owner = operand.getOwner();
            return owner != storeOp && owner != castOp && owner != castBackOp;
          });
    }
  }

  // Returns the indices and the map for the load store from L1 and Dst.
  //   tuple(l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices).
  static std::tuple<AffineMap, SmallVector<Value>, AffineMap,
                    SmallVector<Value>>
  buildIndices(PatternRewriter &rewriter, Location loc,
               const mlir::IRMapping &irMapper, ValueRange currentIndices,
               int64_t dstSliceIndex, AffineMap map) {
    AffineMap l1AccessMap = map;
    SmallVector<Value> l1AccessIndices =
        llvm::to_vector(llvm::map_range(currentIndices, [&](Value index) {
          return irMapper.lookupOrDefault(index);
        }));

    AffineMap dstAccessMap = map.insertResult(
        getAffineConstantExpr(dstSliceIndex, rewriter.getContext()), 0);
    SmallVector<Value> dstAccessIndices = l1AccessIndices;
    return {l1AccessMap, l1AccessIndices, dstAccessMap, dstAccessIndices};
  }

  bool useTileMatmul = false;
  unsigned maxDstPhysicalSizeTiles = 0;
  const DestRegisterAnalysis *analysis = nullptr;
  mutable size_t genericOpIndex = 0;
};
} // namespace

namespace {
template <typename TileReduceOp>
class D2MPackerMaskResetRewriter : public OpRewritePattern<TileReduceOp> {
public:
  using OpRewritePattern<TileReduceOp>::OpRewritePattern;

  Value index(OpBuilder &rewriter, Location loc, int64_t val) const {
    return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexType(),
                                              rewriter.getIndexAttr(val));
  }

  LogicalResult matchAndRewrite(TileReduceOp op,
                                PatternRewriter &rewriter) const final {

    bool packerResetFound = false;
    op->getBlock()->walk([&](Operation *op) {
      if (auto packerReset =
              mlir::dyn_cast_or_null<d2m::PackerMaskResetOp>(op)) {
        packerResetFound = true;
      }
    });
    if (packerResetFound) {
      return failure();
    }

    rewriter.setInsertionPointAfter(op);
    ReduceDim reduceDim = op.getReduceDim();
    SmallVector<int64_t> loopBounds =
        op->template getParentOfType<GenericOp>().getLoopBounds();

    scf::IfOp ifOp;
    if (reduceDim == ReduceDim::R) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::C) {
      auto iterIndex = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndex,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), condOp);
    } else if (reduceDim == ReduceDim::RC) {
      auto iterIndexR = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(1));
      auto iterIndexC = rewriter.create<d2m::IterIndexOp>(
          op.getLoc(), static_cast<int64_t>(0));
      auto condOp = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexR,
          index(rewriter, op.getLoc(), loopBounds[1] - 1));
      auto condOp2 = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, iterIndexC,
          index(rewriter, op.getLoc(), loopBounds[0] - 1));
      auto finalCondOp =
          rewriter.create<arith::OrIOp>(op.getLoc(), condOp, condOp2);
      ifOp = rewriter.create<scf::IfOp>(op.getLoc(), finalCondOp);
    }
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<d2m::PackerMaskResetOp>(op.getLoc());

    return success();
  }
};

} // namespace

namespace {
class D2MInsertDstRegisterAccess
    : public impl::D2MInsertDstRegisterAccessBase<D2MInsertDstRegisterAccess> {
public:
  using impl::D2MInsertDstRegisterAccessBase<
      D2MInsertDstRegisterAccess>::D2MInsertDstRegisterAccessBase;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    // Retrieve the cached analysis from the pass manager instead of recomputing
    auto analysisMaybe = getCachedAnalysis<DestRegisterAnalysis>();
    if (!analysisMaybe.has_value()) {
      llvm::errs() << "ERROR: DestRegisterAnalysis not cached! "
                   << "GenericTileComputeLoops pass must run before "
                   << "InsertDstRegisterAccess.\n";
      return signalPassFailure();
    }
    DestRegisterAnalysis &analysis = analysisMaybe.value().get();

    llvm::errs() << "INFO: Using cached DestRegisterAnalysis from "
                 << "GenericTileComputeLoops pass.\n";

    // Print analysis results
    llvm::errs()
        << "\n=== DestRegisterAnalysis Results (InsertDstRegisterAccess) "
           "===\n";
    // for (auto &[genericOp, dstInfo] : analysis.genericOpMap) {
    //   llvm::errs() << "GenericOp: " << genericOp->getName().getStringRef()
    //                << " (max DST usage: " << dstInfo.dstMaxUsage << ")\n";
    //   llvm::errs() << "  Compute Op Map:\n";
    //   for (auto &[computeOp, dstIndex] : dstInfo.computeOpMap) {
    //     llvm::errs() << "    " << computeOp->getName().getStringRef() << " @
    //     "
    //                  << computeOp << " -> DST slice " << dstIndex << "\n";
    //   }
    // }
    llvm::errs() << "Cached analysis contains "
                 << analysis.dstRegisterInfoList.size() << " GenericOps\n";
    for (size_t i = 0; i < analysis.dstRegisterInfoList.size(); ++i) {
      const auto &dstInfo = analysis.dstRegisterInfoList[i];
      llvm::errs() << "  GenericOp #" << i
                   << " has max DST usage: " << dstInfo.dstMaxUsage << "\n";
      llvm::errs() << "    DST Slice Indices ("
                   << dstInfo.dstSliceIndices.size() << " compute ops): ";
      for (int idx : dstInfo.dstSliceIndices) {
        llvm::errs() << idx << " ";
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "=====================================================\n\n";

    patterns.add<D2MInsertDstRegisterAccessRewriter>(
        ctx, useTileMatmul, maxDstPhysicalSizeTiles.getValue(), &analysis);

    patterns.add<D2MPackerMaskResetRewriter<TileReduceSumOp>,
                 D2MPackerMaskResetRewriter<TileReduceMaxOp>>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt::d2m
