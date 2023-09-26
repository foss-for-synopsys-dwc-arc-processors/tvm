/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/metaware/codegen.cc
 *
 * \brief Implementation of MWTVM codegen APIs.
 *
 * For now only supporting MWTVM CompilationType::HOST_FLOAT
 */

#include <sys/types.h>
#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>

/**
 * @todo make this an external lib.
 */
// #include "../../../../runtime/contrib/metaware/mwtvm.h"

#include <mwtvm/mwtvm.hpp>

#include "../../utils.h"

namespace tvm::relay::contrib {

using namespace backend;

#include <sys/stat.h>

#include <fstream>
#include <iostream>

using namespace snps_arc::metaware::mwtvm;

TVM_REGISTER_GLOBAL("compile.MetaWareSetCompileWorkDirectory")
    .set_body_typed(snps_arc::metaware::mwtvm::SetCompileWorkDirectory);

static CompilationType compile_mode = CompilationType::CALIBRATE;

std::string MetawareSetCompilationMode(std::string mode) {
  if (mode == "calibrate") {
    compile_mode = CompilationType::CALIBRATE;
  } else if (mode == "host_fixed") {
    compile_mode = CompilationType::HOST_FIXED;
  } else if (mode == "unmerged_large") {
    compile_mode = CompilationType::TARGET;
  } else {
    return "MetaWare compilation type `" + mode + "' not supported";
  }

  return "ok";
}

TVM_REGISTER_GLOBAL("compile.MetawareSetCompilationMode")
    .set_body_typed(MetawareSetCompilationMode);

/**
 * \brief Set any extra compile options for external MWTVM compilation.
 */

std::string MetaWareExtraCompileOptions(const std::string& target, const std::string& options) {
  std::string err_message;

  ICHECK(SetExtraCompileOptions(target, options, err_message)) << err_message;

  return "ok";
}

TVM_REGISTER_GLOBAL("compile.MetawareSetCompilationOptions")
    .set_body_typed(MetaWareExtraCompileOptions);

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compiles it into a runtime module.
 *
 * For now only supporting MWTVM CompilationType::HOST_FLOAT
 */
runtime::Module MetaWareCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);

  const auto* cvt_pf = runtime::Registry::Get("relay.ext.convert_to_onnx");
  ICHECK(cvt_pf != nullptr) << "Metaware compiler Cannot find ONNX conversion function";

  std::string onnx_model = (*cvt_pf)(func, "mwtvm_onnx_function");

  // @todo Remove this parameter, they will be in ONNX model?
  //
  Array<String> const_names;

  const auto* pf = runtime::Registry::Get("runtime.MetaWareRuntimeCreate");
  ICHECK(pf != nullptr) << "Metaware compiler Cannot find MetaWare runtime module to create";

  std::string mwtvm_bin, err_message;
  std::string hash;

  int subgraph_number = Compile(onnx_model, mwtvm_bin, hash, err_message, compile_mode);

  ICHECK(subgraph_number >= 0) << "Problems running MWTVM compilation: " << err_message;

  auto mod = (*pf)(func_name, subgraph_number, mwtvm_bin, hash, const_names);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.metaware").set_body_typed(MetaWareCompiler);

/**
 * @brief Print out process ID, and wait for debugger to attach
 *
 * This is so we can get a C++ debugger going while we are also debugging python.
 */

void MetaWareAttachGdb() {
  volatile bool gdb_attached = false;

  while (!gdb_attached) {
    printf("MetaWare waiting for GDB attach, PID = %d\n", (int)getpid());
    ICHECK(system("sleep 1") != -1) << "Problems with sleep command";
  }
}

TVM_REGISTER_GLOBAL("metaware.AttachGdb").set_body_typed(MetaWareAttachGdb);

}  // namespace tvm::relay::contrib
