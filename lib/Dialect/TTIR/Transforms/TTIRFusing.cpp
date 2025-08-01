// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRFUSING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
// Check if we can fuse conv2d followed by add into conv2d with bias.
class TTIRConv2dWithBias : public mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(AddOp srcOp, mlir::PatternRewriter &rewriter) const final {
    auto components = getConv2dAndBias(srcOp);
    if (!components) {
      return mlir::failure();
    }

    auto conv2dOp = components->first;
    auto bias = components->second;
    rewriter.modifyOpInPlace(conv2dOp,
                             [&]() { conv2dOp.getBiasMutable().assign(bias); });
    rewriter.replaceAllOpUsesWith(srcOp, conv2dOp);

    // The original conv2d op will be removed by DCE since it's no longer
    // used.
    return mlir::success();
  }

private:
  bool isFusable(ttir::Conv2dOp conv2dOp,
                 mlir::TypedValue<mlir::RankedTensorType> bias) const {
    return conv2dOp && !conv2dOp.getBias() && conv2dOp->hasOneUse() &&
           conv2dOp.isBiasCompatible(bias.getType().getShape()) &&
           (!bias.getDefiningOp() ||
            bias.getDefiningOp()->isBeforeInBlock(conv2dOp));
  }

  std::optional<std::pair<ttir::Conv2dOp, mlir::Value>>
  getConv2dAndBias(AddOp srcOp) const {
    auto lhs = srcOp.getLhs();
    auto rhs = srcOp.getRhs();
    auto lhsConv2dOp = lhs.getDefiningOp<ttir::Conv2dOp>();
    auto rhsConv2dOp = rhs.getDefiningOp<ttir::Conv2dOp>();
    if (isFusable(lhsConv2dOp, rhs)) {
      return std::make_pair(lhsConv2dOp, rhs);
    }
    if (isFusable(rhsConv2dOp, lhs)) {
      return std::make_pair(rhsConv2dOp, lhs);
    }
    return std::nullopt;
  }
};

// This pattern detects when a reduction operation is followed by a reshape
// operation that simply adds back dimensions that were reduced. In such cases,
// we can fuse the reshape into the reduction operation by setting
// keep_dim=true.
template <typename ReductionOpTy>
class ReductionWithReshapePattern
    : public mlir::OpRewritePattern<ReductionOpTy> {
  using mlir::OpRewritePattern<ReductionOpTy>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(ReductionOpTy reductionOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if the reduction op has exactly one use and it's a reshape op.
    if (!isFusable(reductionOp)) {
      return mlir::failure();
    }

    // Get the reshape op that follows the reduction op.
    auto reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Replce old reduction with new reduction with keep_dim=true.
    utils::replaceOpWithNewDPSOp<ReductionOpTy>(
        rewriter, reductionOp, reshapeOp.getResult().getType(),
        reductionOp.getInput(), /*keep_dim=*/rewriter.getBoolAttr(true),
        reductionOp.getDimArgAttr());

    // Reshape will be folded into the new reduction op.
    return mlir::success();
  }

private:
  bool isFusable(ReductionOpTy reductionOp) const {
    // Reduction should only have one use and that use should be a reshape op.
    if (!reductionOp.getResult().hasOneUse() ||
        !ttmlir::utils::allUsersOfType<ReshapeOp>(reductionOp)) {
      return false;
    }

    // If keep dim is already set, we cannot fuse.
    if (reductionOp.getKeepDim()) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*reductionOp.getResult().getUsers().begin());

    // Get the input and output shapes.
    ArrayRef<int64_t> inputShape = reductionOp.getInput().getType().getShape();
    ArrayRef<int64_t> reshapeOutputShape =
        reshapeOp.getResult().getType().getShape();

    // Reshape output shape should have the same rank as the input shape.
    if (reshapeOutputShape.size() != inputShape.size()) {
      return false;
    }

    // Calculate the expected shape after reduction with keep_dim=true.
    llvm::SmallVector<int64_t> expectedShape(inputShape);

    if (!reductionOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced to 1.
      expectedShape = llvm::SmallVector<int64_t>(inputShape.size(), 1);
    } else {
      // Only specified dimensions are reduced to 1.
      mlir::ArrayAttr reduceDims = *reductionOp.getDimArg();
      for (mlir::Attribute reduceDim : reduceDims) {
        int64_t reduceDimInt =
            mlir::cast<mlir::IntegerAttr>(reduceDim).getInt();
        // Handle negative indices.
        reduceDimInt = (reduceDimInt + inputShape.size()) % inputShape.size();
        expectedShape[reduceDimInt] = 1;
      }
    }

    // Check if the reshape output matches the expected shape.
    return expectedShape == reshapeOutputShape;
  }
};

// This pattern detects and fuses a sequence of operations that implement
// softmax:
// 1. exp(x)
// 2. sum(exp(x)) along a dimension with keep_dim=true
// 3. broadcast the sum
// 4. divide exp(x) by the broadcasted sum
class SoftmaxFusionPattern : public mlir::OpRewritePattern<DivOp> {
  using mlir::OpRewritePattern<DivOp>::OpRewritePattern;

public:
  // Pattern: div(exp(x), broadcast(sum(exp(x), keep_dim=true))).
  mlir::LogicalResult
  matchAndRewrite(DivOp divOp, mlir::PatternRewriter &rewriter) const final {

    // Get the numerator (exp) and denominator (broadcast).
    mlir::Value numerator = divOp.getLhs();
    mlir::Value denominator = divOp.getRhs();

    // Check that the numerator is an exp operation.
    auto expOp = numerator.getDefiningOp<ExpOp>();
    if (!expOp) {
      return mlir::failure();
    }

    // Check that the denominator is a broadcast operation.
    auto broadcastOp = denominator.getDefiningOp<BroadcastOp>();
    if (!broadcastOp) {
      return mlir::failure();
    }

    // Check that the broadcast input is a sum operation with keep_dim=true.
    auto sumOp = broadcastOp.getInput().getDefiningOp<SumOp>();
    if (!sumOp || !sumOp.getKeepDim()) {
      return mlir::failure();
    }

    // Check that the sum input is the same exp operation as the numerator.
    if (sumOp.getInput() != expOp.getResult()) {
      return mlir::failure();
    }

    // Check correct user counts for each operation in the pattern:
    // - exp should have exactly 2 users (sum and div)
    // - sum should have exactly 1 user (broadcast)
    // - broadcast should have exactly 1 user (div)
    // TODO(milant): Relax requirements for fusing #3183.
    if (ttmlir::utils::countUsers(expOp.getResult()) != 2 ||
        !sumOp.getResult().hasOneUse() ||
        !broadcastOp.getResult().hasOneUse()) {
      return mlir::failure();
    }

    // Get the reduction dimension from the sum operation.
    if (!sumOp.getDimArg()) {
      // If no dimensions are specified, all dimensions are reduced, which is
      // not what we want for softmax.
      return mlir::failure();
    }

    mlir::ArrayAttr reduceDims = *sumOp.getDimArg();
    if (reduceDims.size() != 1) {
      // Softmax reduces along a single dimension.
      return mlir::failure();
    }

    int64_t reduceDim = mlir::cast<mlir::IntegerAttr>(reduceDims[0]).getInt();

    // Replace div op with new softmax op.
    utils::replaceOpWithNewDPSOp<SoftmaxOp>(rewriter, divOp,
                                            divOp.getResult().getType(),
                                            expOp.getInput(), reduceDim);

    return mlir::success();
  }
};

class Conv2dWithMultiply : public mlir::OpRewritePattern<MultiplyOp> {
  using mlir::OpRewritePattern<MultiplyOp>::OpRewritePattern;

public:
  /// Pattern: conv2d(input, weight) * scale
  ///
  /// This pattern detects when a Conv2d operation (with constant weights) is
  /// followed by a multiplication with a constant scale factor. It optimizes
  /// this pattern by pre-multiplying the convolution weights with the scale
  /// factor, which eliminates the runtime multiplication operation.
  ///
  /// Input pattern:
  ///   %conv = conv2d(%input, %weight)  // weight is constant
  ///   %result = multiply(%conv, %scale)  // scale is constant
  ///   (1,1,1,out_channels)
  ///
  /// Output pattern:
  ///   %reshaped_scale = reshape(%scale) to (out_channels,1,1,1)
  ///   %scaled_weight = multiply(%weight, %reshaped_scale)
  ///   %result = conv2d(%input, %scaled_weight)
  mlir::LogicalResult
  matchAndRewrite(MultiplyOp multiplyOp,
                  mlir::PatternRewriter &rewriter) const final {
    // Check if this pattern is applicable.
    auto components = getConv2dAndScale(multiplyOp);
    if (!components) {
      return mlir::failure();
    }

    Conv2dOp conv2dOp = components->first;
    Value scaleValue = components->second;

    // Reshape scale to match weight dimensions and pre-multiply weights.
    Value reshapedScale =
        createReshapedScale(rewriter, conv2dOp.getLoc(), scaleValue,
                            conv2dOp.getWeight().getType());

    // Get UD chain starting from the reshaped scale. This chain will be
    // moved before the conv2dOp to ensure that weight scale can be
    // const-evaled.
    SetVector<Value> udChain = ttmlir::utils::getUseDefChain(reshapedScale);
    SetVector<Operation *> udChainOps =
        ttmlir::utils::filterOperations(udChain.getArrayRef());
    SetVector<Operation *> udChainSorted = topologicalSort(udChainOps);
    for (auto *op : udChainSorted) {
      op->moveBefore(conv2dOp);
    }

    rewriter.setInsertionPoint(conv2dOp);

    // Create scaled weights by multiplying the original weights with the
    // resshaped scale.
    Value scaledWeights = createScaledWeights(
        rewriter, conv2dOp.getLoc(), conv2dOp.getWeight(), reshapedScale);

    // Update conv2d to use scaled weights and replace multiply operation.
    rewriter.modifyOpInPlace(
        conv2dOp, [&]() { conv2dOp.getWeightMutable().assign(scaledWeights); });
    rewriter.replaceAllOpUsesWith(multiplyOp, conv2dOp);

    return mlir::success();
  }

private:
  static std::optional<std::pair<Conv2dOp, mlir::Value>>
  getConv2dAndScale(MultiplyOp multiplyOp) {
    auto lhs = multiplyOp.getLhs();
    auto rhs = multiplyOp.getRhs();
    Conv2dOp lhsConv2d = lhs.getDefiningOp<Conv2dOp>();
    Conv2dOp rhsConv2d = rhs.getDefiningOp<Conv2dOp>();
    if (isCommutable(lhsConv2d, rhs)) {
      return std::make_pair(lhsConv2d, rhs);
    }
    if (isCommutable(rhsConv2d, lhs)) {
      return std::make_pair(rhsConv2d, lhs);
    }
    return std::nullopt;
  }

  // We can commute only if both scale and weight are constant.
  static bool isCommutable(Conv2dOp conv2dOp,
                           mlir::TypedValue<RankedTensorType> scale) {
    // Conv2d should only have one use and that use should be a multiply op.
    if (!conv2dOp || !conv2dOp.getResult().hasOneUse()) {
      return false;
    }

    mlir::func::FuncOp funcOp = conv2dOp->getParentOfType<mlir::func::FuncOp>();
    llvm::SmallPtrSet<BlockArgument, 4> constParams =
        mlir::tt::ttcore::getConstsAndParams(funcOp);
    auto isConstant = [&constParams](mlir::Value value) {
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
        return constParams.contains(blockArg);
      }

      Operation *defOp = value.getDefiningOp();
      return defOp->hasTrait<mlir::tt::ttcore::Trait::TTCoreCreationOpTrait>();
    };

    // If weight is not constant, we cannot commute.
    if (!isConstant(conv2dOp.getWeight())) {
      return false;
    }

    RankedTensorType scaleType = scale.getType();
    // If scale is comming from broadcast then we want to use the input type
    // to the broadcast to check the shape.
    if (auto bcastOp =
            mlir::dyn_cast_if_present<BroadcastOp>(scale.getDefiningOp())) {
      scaleType = bcastOp.getInput().getType();
    }

    // Check if scale shape is with conv2d weight.
    if (!hasValidScaleShape(conv2dOp, scaleType)) {
      return false;
    }

    // Now we want to check if operations which produce scale are
    // const-evalable. We do this by getting UD chain of the scale and then
    // checking if all inputs into this chain are constants.
    SetVector<Value> useDefChain = ttmlir::utils::getUseDefChain(scale);
    SetVector<BlockArgument> useDefChainBlockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
    if (!all_of(useDefChainBlockArgs, isConstant)) {
      return false;
    }

    // Since we want to move the scale chain before conv2dOp we want to make
    // sure that the scale chain does not contain conv2dOp.
    if (useDefChain.contains(conv2dOp)) {
      return false;
    }

    return true;
  }

  // Scale must have rank 4 and shape (1, 1, 1, out_channels).
  static bool hasValidScaleShape(Conv2dOp convOp, RankedTensorType scaleType) {
    return scaleType.getRank() == 4 && scaleType.getDimSize(0) == 1 &&
           scaleType.getDimSize(1) == 1 && scaleType.getDimSize(2) == 1 &&
           scaleType.getDimSize(3) == convOp.getOutputChannelSize();
  }

  // There are two cases we want to handle here:
  // 1. Input scale is a constant tensor that only neeeds reshaping
  // 2. Input scale is a broadcast operation that needs reshaping
  //
  // In case of 1 we just add reshape operation to the scale tensor such that
  // it has shape (out_channels, 1, 1, 1).
  //
  // In case of 2 we need to add reshape operation to the input of the of bcast
  // and then we create new broadcast operation with the new reshaped scale
  // which broadcasts the reshaped scale to the shape of the weight tensor.
  static Value createReshapedScale(mlir::PatternRewriter &rewriter,
                                   Location loc, Value scaleValue,
                                   RankedTensorType weightType) {
    // If scaleValue is broadcast operation we want to reshape its input.
    // Otherwise we reshape the scaleValue itself.
    Value reshapeInput = scaleValue;
    if (auto bcastOp = mlir::dyn_cast_if_present<BroadcastOp>(
            scaleValue.getDefiningOp())) {
      rewriter.setInsertionPoint(bcastOp);
      reshapeInput = bcastOp.getInput();
    }

    // Get the scale's type.
    RankedTensorType scaleType =
        mlir::cast<RankedTensorType>(reshapeInput.getType());

    // Create a new shape (out_channels, 1, 1, 1) from (1, 1, 1, out_channels).
    llvm::SmallVector<int64_t> newShape(scaleType.getShape());
    // Swap first and last dimensions.
    assert(newShape.size() == 4 &&
           "Scale tensor must have 4 dimensions for reshaping.");
    std::swap(newShape[0], newShape[3]);
    // Convert to int32 for the reshape operation.
    llvm::SmallVector<int32_t> newShapeI32(newShape.begin(), newShape.end());

    // Create the reshape operation.
    auto reshapedScale = ttir::utils::createDPSOp<ttir::ReshapeOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_reshape"),
        newShape, scaleType.getElementType(), scaleType.getEncoding(),
        reshapeInput, rewriter.getI32ArrayAttr(newShapeI32));

    // If scale value is not a broadcast operation we can return reshapedScale.
    if (!isa_and_present<ttir::BroadcastOp>(scaleValue.getDefiningOp())) {
      return reshapedScale;
    }

    // Otherwise we need to create a new broadcast operation that will take
    // reshaped scale and brroadcast it to the shape of the weight tensor.
    SmallVector<int64_t> broadcastDims =
        ttmlir::utils::getBroadcastDimensions<int64_t>(
            reshapedScale.getType().getShape(), weightType.getShape());
    return ttir::utils::createDPSOp<ttir::BroadcastOp>(
        rewriter, scaleValue.getLoc(), weightType, reshapedScale,
        broadcastDims);
  }

  /// Create pre-multiplied weights.
  static Value createScaledWeights(mlir::PatternRewriter &rewriter,
                                   Location loc, Value weightValue,
                                   Value reshapedScale) {
    // Create a multiplication of the weights by the reshaped scale.
    return utils::createDPSOp<MultiplyOp>(
        rewriter, ttmlir::utils::appendLocationSuffix(loc, "_multiply"),
        mlir::cast<RankedTensorType>(weightValue.getType()), weightValue,
        reshapedScale);
  }
};

// Tag all block arguments which are direct inputs to Conv2dOp with
// discardable attribute. This is used during Layouting to check if
// function argument need to be put to Host/RM. This is temporary
// solution until we complete refactor of TTNNLayout see:
// https://github.com/tenstorrent/tt-mlir/issues/3432.
class Conv2dTagWeights : public mlir::OpRewritePattern<Conv2dOp> {
public:
  Conv2dTagWeights(MLIRContext *context)
      : OpRewritePattern(context, PatternBenefit(2)) {}

  mlir::LogicalResult
  matchAndRewrite(Conv2dOp conv2d,
                  mlir::PatternRewriter &rewriter) const final {
    if (BlockArgument blockArg = dyn_cast<BlockArgument>(conv2d.getWeight())) {
      // Get the function that owns this block argument.
      func::FuncOp owningFunc =
          cast<func::FuncOp>(blockArg.getOwner()->getParentOp());

      // Get the argument index.
      uint32_t argIdx = blockArg.getArgNumber();

      // Check if the argument already has the g_conv2dWeight attribute.
      if (owningFunc.getArgAttr(argIdx,
                                ttmlir::utils::g_conv2dWeightAttrName)) {
        return failure();
      }

      // Create the g_conv2dWeight attribute.
      auto gConv2dWeightAttr = rewriter.getUnitAttr();
      owningFunc.setArgAttr(argIdx, ttmlir::utils::g_conv2dWeightAttrName,
                            gConv2dWeightAttr);

      return success();
    }

    return failure();
  }
};

class CacheFillUpdatePattern : public mlir::OpRewritePattern<ScatterOp> {
  using mlir::OpRewritePattern<ScatterOp>::OpRewritePattern;

public:
  /// Pattern: scatter(input, indices, updates)
  ///
  /// This pattern detects when a ScatterOp is used as a fill/update for a
  /// cache. We check for its input, indices, and update tensors to ensure they
  /// match the expected cache fill/update pattern.
  ///
  /// Input pattern:
  ///   %result = scatter(%cache, %indices, %updates)
  ///   - Given a cache with shape (B, N, M, H) and a updates tensor with shape
  ///   (B, N, S, H), the indices tensor represents the index where each element
  ///   in %updates should placed in the %cache.
  ///   - %indices can be tracked back to the function's cachePositions input
  ///   that represents the indices of the cache to fill/update.
  /// Output pattern:
  ///   %result = fillCacheOp(%cache, %updates)
  ///   or (if S == 1)
  ///   %result = updateCacheOp(%cache, %updates, %update_index)
  mlir::LogicalResult
  matchAndRewrite(ScatterOp scatterOp,
                  mlir::PatternRewriter &rewriter) const final {
    auto CachePositions = getCacheUpdatePositions(scatterOp);
    if (!CachePositions) {
      return mlir::failure();
    }

    auto cacheUpdateInputType =
        mlir::cast<RankedTensorType>((*CachePositions).getType());
    auto cacheUpdateInputShape = cacheUpdateInputType.getShape();
    if (cacheUpdateInputShape.size() != 1) {
      return mlir::failure();
    }

    Value cache = scatterOp.getInput();
    Value updates = scatterOp.getUpdate();
    auto batchOffsetAttr = rewriter.getI32IntegerAttr(0);

    // If the cachePositions tensor has more than one element we assume it
    // represents a set of aranged indices (0, cachePositions.size), so we
    // replace it with FillCacheOp. If the tensor has only one element, we
    // assume it represents the update index for UpateCacheOp.
    if (cacheUpdateInputShape[0] != 1) {
      rewriter.replaceOpWithNewOp<FillCacheOp>(
          scatterOp, scatterOp.getResult().getType(), // Result type
          cache,                                      // Cache tensor
          updates,                                    // Updates tensor
          batchOffsetAttr                             // Batch offset
      );
    } else {
      rewriter.replaceOpWithNewOp<UpdateCacheOp>(
          scatterOp, scatterOp.getResult().getType(), // Result type
          cache,                                      // Cache tensor
          updates,                                    // Updates tensor
          *CachePositions,                            // Cache Idx
          batchOffsetAttr                             // Batch offset
      );
    }

    return mlir::success();
  }

private:
  // Check if the scatter op is a cache fill/update, and track the
  // cachePositions input tensor if it is.
  //
  // We are looking for:
  // %result = "ttir.scatter"(%cache, %indices, %updates)
  // Where:
  //    1. %cache and %updates are 4D tensors who's shape match except on the
  //    2nd dimension,
  //       (B, N, M, H) and (B, N, S, H) respectively, M being the max cache
  //       length and S being the sequence length of the update.
  //    2. %indices comes from a block argument representing the cachePositions
  //    tensor.
  static std::optional<mlir::Value>
  getCacheUpdatePositions(ttir::ScatterOp scatterOp) {
    // Check that the scatter op inputs represent a cache fill/update:
    //    1. The input is a 4D (B, N, M, H)
    //    2. The update tensor is a 4D tensor (B, N, S, H)
    //    3. The scatter indices is either a 1D equivalent tensor or 5D index
    //       grid tensor (B, N, S, H, 4). Both can be tracked to a block
    //       argument representing the cachePositions input.
    auto scatterIndices = scatterOp.getScatterIndices();
    ArrayRef<int64_t> inputShape =
        mlir::cast<RankedTensorType>(scatterOp.getInput().getType()).getShape();
    ArrayRef<int64_t> scatterIdxShape =
        mlir::cast<RankedTensorType>(scatterIndices.getType()).getShape();
    ArrayRef<int64_t> updateShape =
        mlir::cast<RankedTensorType>(scatterOp.getUpdate().getType())
            .getShape();
    if (inputShape.size() != 4 || updateShape.size() != 4) {
      return std::nullopt;
    }

    if (!(inputShape[0] == updateShape[0] && inputShape[1] == updateShape[1] &&
          inputShape[3] == updateShape[3])) {
      return std::nullopt;
    }

    int cacheUpdateSize = updateShape[2];

    int64_t effectiveSize = 1;
    bool effectively1D = isEffectively1D(scatterIdxShape, effectiveSize);
    if (effectively1D && effectiveSize != cacheUpdateSize) {
      return std::nullopt;
    }

    bool isIndexGrid =
        (scatterIdxShape.size() == 5 && scatterIdxShape[0] == inputShape[0] &&
         scatterIdxShape[1] == inputShape[1] &&
         scatterIdxShape[2] == cacheUpdateSize &&
         scatterIdxShape[3] == inputShape[3] && scatterIdxShape[4] == 4);

    // Check that scatter indices is either a 1D cache positions tensor or a 5D
    // index grid.
    if (!effectively1D && !isIndexGrid) {
      return std::nullopt;
    }

    // The cachePositions tensor is expected to be a 1D blockargument tensor
    // with the same size as the cache update size.
    auto useDefChain = ttmlir::utils::getUseDefChain(scatterIndices);
    auto blockArgs =
        ttmlir::utils::filterBlockArguments(useDefChain.getArrayRef());
    for (auto blockArg : blockArgs) {
      // Check if the block argument is a cachePositions input.
      auto argTensorShape =
          mlir::cast<RankedTensorType>(blockArg.getType()).getShape();
      effectively1D = isEffectively1D(argTensorShape, effectiveSize);
      if (!effectively1D) {
        continue;
      }
      if (effectiveSize == cacheUpdateSize) {
        // We found the cachePositions input tensor.
        return blockArg;
      }
    }

    return std::nullopt;
  }

  static bool isEffectively1D(ArrayRef<int64_t> &shape,
                              int64_t &effectiveSize) {
    effectiveSize = 1;
    int64_t nonUnaryDims = 0;
    for (int64_t dim : shape) {
      if (dim != 1) {
        nonUnaryDims++;
        effectiveSize *= dim;
      }
    }
    return nonUnaryDims <= 1;
  }
};

class TTIRFusingPass : public impl::TTIRFusingBase<TTIRFusingPass> {
public:
  using impl::TTIRFusingBase<TTIRFusingPass>::TTIRFusingBase;
  void runOnOperation() final {
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<Conv2dTagWeights>(&getContext());
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRConv2dWithBias>(&getContext());

      // Add patterns for each reduction op type.
      patterns.add<ReductionWithReshapePattern<SumOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<MeanOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<MaxOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<MinOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<ProdOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<ReduceAndOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<ReduceOrOp>>(&getContext());
      patterns.add<ReductionWithReshapePattern<ArgMaxOp>>(&getContext());

      patterns.add<SoftmaxFusionPattern>(&getContext());
      patterns.add<Conv2dWithMultiply>(&getContext());
      patterns.add<CacheFillUpdatePattern>(&getContext());

      GreedyRewriteConfig config;
      config.setUseTopDownTraversal(true);
      (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir
