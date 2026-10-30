// Isolate the historical async marker attribute omission after LLVM's parser
// has upgraded intrinsic declarations to the currently compiled definitions.
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"

using namespace llvm;

struct StripMarkerNoCallback : RequiredPassInfoMixin<StripMarkerNoCallback> {
  PreservedAnalyses run(Module &module, ModuleAnalysisManager &) {
    for (const char *name : {"llvm.amdgcn.asyncmark", "llvm.amdgcn.wait.asyncmark"})
      if (Function *function = module.getFunction(name))
        function->removeFnAttr(Attribute::NoCallback);
    return PreservedAnalyses::none();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "strip-marker-nocallback", LLVM_VERSION_STRING,
          [](PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](StringRef name, ModulePassManager &pm,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (name != "strip-marker-nocallback")
                    return false;
                  pm.addPass(StripMarkerNoCallback());
                  return true;
                });
          }};
}
