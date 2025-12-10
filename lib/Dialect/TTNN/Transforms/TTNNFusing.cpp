// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNFUSING
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

template <typename ActivationOp>
class TTNNConv2dWithActivation : public mlir::OpRewritePattern<Conv2dOp> {
  using TTNNConv2dWithActivation::OpRewritePattern<Conv2dOp>::OpRewritePattern;

public:
  mlir::LogicalResult
  matchAndRewrite(Conv2dOp srcOp, mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp = getActivationOp(srcOp);
    Value activationInput = activationOp.getInput();

    auto activation = getActivationOpType(rewriter);

    // For conv2d that will be converted to matmul, we cannot fuse relu6 as
    // tt-metal doesn't support relu6 fusion for matmul.
    // TODO(mvasiljevic): remove this once tt-metal supports relu6 fusion for
    // matmul (https://github.com/tenstorrent/tt-metal/issues/29886)
    if (isMatmul(srcOp) && activation == ttnn::UnaryOpType::Relu6) {
      return failure();
    }

    ttcore::DataType weightDtype = ttcore::elementTypeToDataType(
        srcOp.getWeight().getType().getElementType());
    Conv2dConfigAttr conv2dConfigAttr =
        srcOp.getConv2dConfigAttr()
            ? srcOp.getConv2dConfigAttr()
            : Conv2dConfigAttr::get(rewriter.getContext());
    conv2dConfigAttr = conv2dConfigAttr.withActivation(activation)
                           .withWeightsDtype(weightDtype);

    rewriter.modifyOpInPlace(
        srcOp, [&]() { srcOp.setConv2dConfigAttr(conv2dConfigAttr); });

    // Replace the activation op uses with either conv2d or reshape
    // depending on if reshape was present.
    rewriter.replaceAllUsesWith(activationOp, activationInput);

    return mlir::success();
  }

private:
  ActivationOp getActivationOp(Conv2dOp srcOp) const {
    assert((ttmlir::utils::allUsersOfType<ReshapeOp, ActivationOp>(srcOp)) &&
           "Conv2d should have either activation or Reshape as user.");

    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    assert(reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp) &&
           "Reshape should have only one user and that user should be "
           "activation.");
    return mlir::cast<ActivationOp>(*reshapeOp.getResult().getUsers().begin());
  }

  ttnn::UnaryOpType getActivationOpType(mlir::PatternRewriter &rewriter) const {
    if constexpr (std::is_same_v<ActivationOp, ReluOp>) {
      return ttnn::UnaryOpType::Relu;
    } else if constexpr (std::is_same_v<ActivationOp, Relu6Op>) {
      return ttnn::UnaryOpType::Relu6;
    } else if constexpr (std::is_same_v<ActivationOp, SiluOp>) {
      return ttnn::UnaryOpType::Silu;
    } else {
      static_assert(ttmlir::utils::always_false<ActivationOp>(),
                    "Unsupported activation op");
    }
  }

  bool isFusable(Conv2dOp srcOp) const {
    if (srcOp.getConv2dConfig() && srcOp.getConv2dConfig()->hasActivation()) {
      return false;
    }

    // Conv2d has multiple uses so we cannot fuse.
    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    // Conv2d only user is activation so we can fuse.
    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }

    // Since window flattening will add rehape after conv we need to check
    // if there is reshape right after conv2d.
    if (!ttmlir::utils::allUsersOfType<ReshapeOp>(srcOp)) {
      return false;
    }

    ReshapeOp reshapeOp =
        mlir::cast<ReshapeOp>(*srcOp.getResult().getUsers().begin());

    // If we want to fuse activation to conv we need to make sure that reshape
    // has only one user and that user is activation.
    return reshapeOp.getResult().hasOneUse() &&
           ttmlir::utils::allUsersOfType<ActivationOp>(reshapeOp);
  }

  bool isMatmul(Conv2dOp srcOp) const {
    RankedTensorType weightType =
        mlir::cast<RankedTensorType>(srcOp.getWeight().getType());
    llvm::ArrayRef<int64_t> weightShape = weightType.getShape();

    // Check kernel size is 1x1.
    if (weightShape[2] != 1 || weightShape[3] != 1) {
      return false;
    }

    // Check stride is 1.
    llvm::ArrayRef<int32_t> strideAttr = srcOp.getStride();
    if (strideAttr.size() != 2 || strideAttr[0] != 1 || strideAttr[1] != 1) {
      return false;
    }

    // Check padding is 0.
    llvm::ArrayRef<int32_t> paddingAttr = srcOp.getPadding();
    if (llvm::any_of(paddingAttr, [](int32_t v) { return v != 0; })) {
      return false;
    }

    // only groups = 1 is supported for now
    if (srcOp.getGroups() != 1) {
      return false;
    }
    return true;
  }
};

template <typename SrcOp, typename ActivationOp>
class TTNNEltwiseBinaryWithActivation  : public mlir::OpRewritePattern<SrcOp> {
  using TTNNEltwiseBinaryWithActivation::template OpRewritePattern<SrcOp>::OpRewritePattern;

public:
  mlir::LogicalResult matchAndRewrite(SrcOp srcOp,
                                      mlir::PatternRewriter &rewriter) const final {
    if (!isFusable(srcOp)) {
      return failure();
    }

    ActivationOp activationOp = getActivationOp(srcOp);
    Value activationInput = activationOp.getInput();

    ttnn::UnaryOpType activation = getActivationOpType(rewriter);
    
    // Create new config attributes array
    llvm::SmallVector<mlir::Attribute> newActivationAttrs;

    // Copy existing list of activation attributes if any
    if (auto actAttr = srcOp.getActivation()) {
      mlir::ArrayAttr activationAttrs = actAttr.value();  
      for (auto attr : activationAttrs) {
        newActivationAttrs.push_back(attr);
      }
    }

    // create a UnaryWithParamAttr with no params for 'activation'
    auto makeUnaryWithParamAttr = [&](ttnn::UnaryOpType t) -> mlir::Attribute {
      llvm::ArrayRef<FloatAttr> params = llvm::ArrayRef<FloatAttr>{};
      return ttnn::UnaryWithParamAttr::get(rewriter.getContext(), t,
                                           params);
    };

    newActivationAttrs.push_back(makeUnaryWithParamAttr(activation));
    
    // Create UnaryWithParamArrayAttr from elements and set it on the op
    mlir::ArrayAttr replace =
        mlir::ArrayAttr::get(rewriter.getContext(),
                                           newActivationAttrs);
    rewriter.modifyOpInPlace(srcOp, [&]() { srcOp.setActivationAttr(replace); });

    rewriter.replaceAllUsesWith(activationOp, activationInput);
    return mlir::success();
  }

private:
  ActivationOp getActivationOp(SrcOp srcOp) const {  
    assert(ttmlir::utils::allUsersOfType<ActivationOp>(srcOp) &&   
          "Expected activation as only user");  
    return mlir::cast<ActivationOp>(*srcOp.getResult().getUsers().begin());  
  }

  ttnn::UnaryOpType getActivationOpType(mlir::PatternRewriter &rewriter) const {
    if constexpr (std::is_same_v<ActivationOp, ReluOp>) {
      return ttnn::UnaryOpType::Relu;
    } else if constexpr (std::is_same_v<ActivationOp, Relu6Op>) {
      return ttnn::UnaryOpType::Relu6;
    } else if constexpr (std::is_same_v<ActivationOp, SiluOp>) {
      return ttnn::UnaryOpType::Silu;
    } else if constexpr (std::is_same_v<ActivationOp, SigmoidOp>) {
      return ttnn::UnaryOpType::Sigmoid;
    } else {
      static_assert(ttmlir::utils::always_false<ActivationOp>(),
                    "Unsupported activation op");
    }
  }

  bool isFusable(SrcOp srcOp) const {
    // Check if the operation has multiple uses so we cannot fuse.
    if (!srcOp.getResult().hasOneUse()) {
      return false;
    }

    // Check user is activation so we can fuse.
    if (ttmlir::utils::allUsersOfType<ActivationOp>(srcOp)) {
      return true;
    }
    return false;
  }
};
class TTNNFusingPass : public impl::TTNNFusingBase<TTNNFusingPass> {
public:
  using impl::TTNNFusingBase<TTNNFusingPass>::TTNNFusingBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTNNConv2dWithActivation<ReluOp>,
                 TTNNConv2dWithActivation<Relu6Op>,
                 TTNNConv2dWithActivation<SiluOp>,
                 TTNNEltwiseBinaryWithActivation<AddOp, ReluOp>,
                 TTNNEltwiseBinaryWithActivation<AddOp, Relu6Op>,
                 TTNNEltwiseBinaryWithActivation<AddOp, SiluOp>,
                 TTNNEltwiseBinaryWithActivation<AddOp, SigmoidOp>,
                 TTNNEltwiseBinaryWithActivation<DivideOp, ReluOp>,
                 TTNNEltwiseBinaryWithActivation<DivideOp, Relu6Op>,
                 TTNNEltwiseBinaryWithActivation<DivideOp, SiluOp>,
                 TTNNEltwiseBinaryWithActivation<DivideOp, SigmoidOp>,
                 TTNNEltwiseBinaryWithActivation<SubtractOp, ReluOp>,
                 TTNNEltwiseBinaryWithActivation<SubtractOp, Relu6Op>,
                 TTNNEltwiseBinaryWithActivation<SubtractOp, SiluOp>,
                 TTNNEltwiseBinaryWithActivation<SubtractOp, SigmoidOp>,
                 TTNNEltwiseBinaryWithActivation<MultiplyOp, ReluOp>,
                 TTNNEltwiseBinaryWithActivation<MultiplyOp, Relu6Op>,
                 TTNNEltwiseBinaryWithActivation<MultiplyOp, SiluOp>,
                 TTNNEltwiseBinaryWithActivation<MultiplyOp, SigmoidOp>>(&getContext());
    
    // // Define operation and activation types  
    // using EltwiseBinaryOps = std::tuple<AddOp, DivideOp, SubtractOp, MultiplyOp>;  
    // using ActivationOps = std::tuple<ReluOp, Relu6Op, SiluOp, SigmoidOp>;  
      
    // // For loop to add all combinations  
    // auto addPatterns = [&](auto eltwiseOp, auto ActivationOp) {
    //     using OpE = std::decay_t<decltype(eltwiseOp)>;
    //     using OpA = std::decay_t<decltype(ActivationOp)>;

    //     patterns.add<TTNNEltwiseBinaryWithActivation<OpE, OpA>>(&getContext());
    // };

    // std::apply([&](auto ...Es) {
    //     std::apply([&](auto ...As) {
    //         (addPatterns(Es, As), ...);
    //     }, ActivationOps{});
    // }, EltwiseBinaryOps{});
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};
} // namespace

} // namespace mlir::tt::ttnn
