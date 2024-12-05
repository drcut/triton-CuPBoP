#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value numPrograms;
    if (numCTAs == 1) {
      switch (op.getAxisAsInt()) {
      case 0:
        numPrograms =
            rewriter.create<NVVM::GridDimXOp>(loc, rewriter.getI32Type());
        break;
      case 1:
        numPrograms =
            rewriter.create<NVVM::GridDimYOp>(loc, rewriter.getI32Type());
        break;
      case 2:
        numPrograms =
            rewriter.create<NVVM::GridDimZOp>(loc, rewriter.getI32Type());
        break;
      }
    } else {
      std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
      sreg.append(1, 'x' + op.getAxisAsInt()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'
      numPrograms = LLVM::NVIDIA::getSRegValue(rewriter, loc, sreg);
    }
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
