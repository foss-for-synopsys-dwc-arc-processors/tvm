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
 * \file src/runtime/contrib/metaware/metaware_runtime.cc
 *
 * \brief A simple runtime for METAWARE.
 *
 * At the moment, we are only supporting host-side emulation,
 * so actual execution on target hardware is not supported.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include <mwtvm/mwtvm.hpp>

namespace tvm::runtime::contrib::metaware {

using snps_arc::metaware::mwtvm::SubgraphExecutionFunction;

class MetaWareLoaderBase : public ModuleNode {
 public:
  template <typename T,
            typename = typename std::enable_if<std::is_base_of<MetaWareLoaderBase, T>::value>::type>
  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol;
    int subgraph_number;
    std::string graph_bin;
    std::vector<std::string> consts;

    // Load the symbol
    ICHECK(stream->Read(&symbol)) << "Loading symbol name failed";
    ICHECK(stream->Read(&subgraph_number)) << "Loading subgraph_number  failed";
    ICHECK(stream->Read(&graph_bin)) << "Loading graph json failed";
    ICHECK(stream->Read(&consts)) << "Loading the const name list failed";

    Array<String> const_names;
    for (const auto& it : consts) {
      const_names.push_back(it);
    }

    // printf("\n\n*** Load of binary with subgraph = %d!\n\n", subgraph_number);

    // printf("MetaWareLoaderBase: Loading module from Binary of %d bytes...\n",
    //        (int)graph_bin.size());

    auto n = make_object<T>(symbol, subgraph_number, graph_bin, const_names);
    return Module(n);
  }

  void SaveToBinary(dmlc::Stream* stream) override {
    stream->Write(symbol_name_);

    stream->Write(subgraph_number_);

    stream->Write(mwtvm_binary);

    std::vector<std::string> consts;
    for (const auto& it : const_names_) {
      consts.push_back(it);
    }

    stream->Write(consts);
  }

 protected:
  /*! \brief The only subgraph name for this module. */
  std::string symbol_name_;

  int subgraph_number_;

  std::string mwtvm_binary;

  /*! \brief The required constant names. */
  Array<String> const_names_;
};

/**
 * Runtime lib.  This version is for host_fixed or host_float execution for now.
 *
 * @todo add runtime variants for various MetaWare runtimes.
 */

class MetaWareRuntime : public MetaWareLoaderBase {
  /**
   * The subgraph execution function is set by the Init method.
   */
  SubgraphExecutionFunction ExecuteSubgraph = nullptr;

 public:
  MetaWareRuntime(const std::string& symbol_name, int subgraph_number, String bin_data,
                  const Array<String> const_names) {
    subgraph_number_ = subgraph_number;

    mwtvm_binary = bin_data;

    this->symbol_name_ = symbol_name;
  }

  const char* type_key() const override { return "metaware_runtime"; }

  void Init(const Array<NDArray>& consts);

  /* Unused stub implementation */
  void Run() {
    printf("TODO: Unused stub implementation should not be used?\n");
    LOG(FATAL) << "Unreachable code";
  }

  /* Thread safe implementation of Run. Keep runtime instance immutable */
  void Run(const TVMArgs& args) const {
    auto extract_dl_tensor = [](const TVMArgValue& val) -> const DLTensor* {
      ICHECK(val.type_code() == kTVMNDArrayHandle || val.type_code() == kTVMDLTensorHandle)
          << "Expect NDArray or DLTensor";
      return val.IsObjectRef<NDArray>() ? val.operator NDArray().operator->()
                                        : val.operator DLTensor*();
    };

    int input_size = 0, output_size = 0;

    auto ChkTensorSize = [](const DLTensor& v) -> int {
      int size = 1;

      ICHECK(v.dtype.code == kTVMArgFloat) << "MetaWare Expecting floating point I/O arguments.";

      for (int i = 0; i < v.ndim; i++) size *= v.shape[i];

      return size;
    };

    /**
     * For now, assume only 1 output, all other args are input
     */

    for (int i = 0; i < args.size() - 1; i++) {
      const DLTensor* v = extract_dl_tensor(args[i]);
      input_size += ChkTensorSize(*v);
    }

    for (int i = args.size() - 1; i < args.size(); i++) {
      const DLTensor* v = extract_dl_tensor(args[i]);
      output_size += ChkTensorSize(*v);
    }

    ICHECK((input_size > 0) && (output_size > 0)) << "Expect non-zero I/O sizes";

    std::vector<float> inputs, outputs(output_size);

    /**
     * The host_fixed/float subgraph implemention function requires contiguous data.
     * Copying data is not efficient for a target execution, so execution on a real
     * target would do this differently (and also perhaps pass fixed point data types).
     */

    for (int i = 0; i < args.size() - 1; i++) {
      const DLTensor* v = extract_dl_tensor(args[i]);
      int sz = ChkTensorSize(*v);
      float* idata = (float*)v->data;
      inputs.insert(inputs.end(), idata, idata + sz);
    }

    // printf("\n\n** MW RUN CALLING THE DL FUNCTION!\n\n");

    int rv = ExecuteSubgraph(inputs.data(), (int)input_size * sizeof(float), outputs.data(),
                             (int)output_size * sizeof(float), nullptr, nullptr);

    ICHECK(rv == 0) << "Issue during execution of MetaWare subgraph implementation "
                    << subgraph_number_ << ", return value of '" << rv << "'";

    /**
     * Copy compute results back to TVM tensor.
     */
    for (int i = args.size() - 1; i < args.size(); i++) {
      const DLTensor* v = extract_dl_tensor(args[i]);
      memcpy(v->data, outputs.data(), output_size * sizeof(float));
    }
  }

  /* Override GetFunction to reimplement Run method */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) override {
    // printf("MetaWare Runtime: Calling GetFunction, name == '%s', symbol_name == '%s'!\n",
    //        name.c_str(), symbol_name_.c_str());

    if (this->symbol_name_ == name) {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        // ICHECK_EQ(args.size(), input_var_eid_.size() + outputs_.size())
        //     << "Found mismatch in the number of provided data entries and required.";

        Run(args);
      });
    } else if (name == "get_symbol") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
    } else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1U);
        std::lock_guard<std::mutex> guard(this->initialize_mutex_);
        if (!this->initialized_) {
          this->Init(args[0]);
          this->initialized_ = true;
        }
        *rv = 0;
      });
    } else {
      // printf("MetaWare Runtime: unknown name!\n");
      return PackedFunc(nullptr);
    }
  }

 private:
  bool initialized_{false};

  /*! \brief Initializer mutex*/
  std::mutex initialize_mutex_;

  // Build up the engine based on the input graph.
  void BuildEngine() { printf("TODO: Implement BuildEngine...\n"); }
};

/**
 * @brief Set the MetaWare working directory
 *
 * This directory is where we can process runtime artifacts.
 */

TVM_REGISTER_GLOBAL("metaware.set_work_directory")
    .set_body_typed(snps_arc::metaware::mwtvm::SetWorkDirectory);

void MetaWareRuntime::Init(const Array<NDArray>& consts) {
  std::string err_message;
  ExecuteSubgraph = snps_arc::metaware::mwtvm::Init(subgraph_number_, mwtvm_binary, err_message);

  ICHECK(ExecuteSubgraph != nullptr) << err_message;
}

runtime::Module MetaWareRuntimeCreate(String symbol_name, int subgraph_number, std::string bin_data,
                                      const Array<String>& const_names) {
  auto n = make_object<MetaWareRuntime>(symbol_name, subgraph_number, bin_data, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MetaWareRuntimeCreate").set_body_typed(MetaWareRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metaware_runtime")
    .set_body_typed(MetaWareLoaderBase::LoadFromBinary<MetaWareRuntime>);

}  // namespace tvm::runtime::contrib::metaware
