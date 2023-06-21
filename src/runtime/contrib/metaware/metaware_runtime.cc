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
#include <mwtvm/mwtvm.hpp>
#include <regex>
#include <string>
#include <vector>

namespace tvm::runtime::contrib::metaware {

namespace mwtvm = snps_arc::metaware::mwtvm;

/**
 * Get a new subgraph number that is unique for this process instance.
 *
 * Note that a particular subgraph may have different numbers at compile time
 * v.s. run time.
 */
int GetNewSubgraphNumber() {
  static int subgraph_number = 0;
  return subgraph_number++;
}

class MetaWareLoaderBase : public ModuleNode {
 public:
  template <typename T,
            typename = typename std::enable_if<std::is_base_of<MetaWareLoaderBase, T>::value>::type>
  static Module LoadFromBinary(void* strm) {
    dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
    std::string symbol;
    int compile_subgraph_number;
    std::string graph_bin;
    std::vector<std::string> consts;

    std::string hash;

    int subgraph_number = GetNewSubgraphNumber();

    // Load the symbol
    ICHECK(stream->Read(&symbol)) << "Loading symbol name failed";
    ICHECK(stream->Read(&hash)) << "Loading hash failed";
    ICHECK(stream->Read(&compile_subgraph_number)) << "Loading subgraph_number  failed";
    ICHECK(stream->Read(&graph_bin)) << "Loading graph json failed";
    ICHECK(stream->Read(&consts)) << "Loading the const name list failed";

    Array<String> const_names;
    for (const auto& it : consts) {
      const_names.push_back(it);
    }

    auto n = make_object<T>(symbol, subgraph_number, compile_subgraph_number, graph_bin, hash,
                            const_names);
    return Module(n);
  }

  void SaveToBinary(dmlc::Stream* stream) override {
    stream->Write(symbol_name_);
    stream->Write(hash_);

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

  std::string hash_;

  int subgraph_number_;
  int compile_subgraph_number_;

  mwtvm::GraphDescriptor graph_descriptor_;

  std::string mwtvm_binary;

  /*! \brief The required constant names. */
  Array<String> const_names_;
};

/**
 * Information on target compilation type,
 */

static std::string calibration_path;

std::string MetaWareSetCalibrationDirectory(String path) {
  calibration_path = path;
  std::string err_msg;

  ICHECK(mwtvm::SetCalibrationBaseDirectory(calibration_path, err_msg))
      << "Issue setting calibration directory: " << err_msg;

  return "ok";
}

TVM_REGISTER_GLOBAL("runtime.MetaWareSetCalibrationDirectory")
    .set_body_typed(MetaWareSetCalibrationDirectory);

/**
 * Runtime lib.  This version is for host_fixed or calibration execution for now.
 *
 * @todo add runtime variants for various MetaWare runtimes.
 */

class MetaWareRuntime : public MetaWareLoaderBase {
 protected:
  mwtvm::GraphInformation graph_info_;

 public:
  MetaWareRuntime(const std::string& symbol_name, int subgraph_number, int compile_subgraph_number,
                  String bin_data, String hash, const Array<String> const_names) {
    hash_ = std::move(hash);
    subgraph_number_ = subgraph_number;
    compile_subgraph_number_ = compile_subgraph_number;

    mwtvm_binary = std::move(bin_data);

    this->symbol_name_ = symbol_name;
  }

  const char* type_key() const override { return "metaware_runtime"; }

  void Init(const Array<NDArray>& consts);

  /* Unused stub implementation */
  void Run() { LOG(FATAL) << "Unreachable code"; }

  /**
   *  Execute subgraph with given I/O arguments.
   */
  void Run(const TVMArgs& args) const {
    /**
     * Get DLTensor argument
     */
    auto ExtractTensor = [](const TVMArgValue& val) -> const DLTensor* {
      ICHECK(val.type_code() == kTVMNDArrayHandle || val.type_code() == kTVMDLTensorHandle)
          << "Expect NDArray or DLTensor";
      return val.IsObjectRef<NDArray>() ? val.operator NDArray().operator->()
                                        : val.operator DLTensor*();
    };

    /**
     * Extract I/O data information into the MWTVM IOTensor data structure.
     */
    auto SetTensorIO = [ExtractTensor, args](int idx, mwtvm::IOTensor& t) {
      const DLTensor& v = *ExtractTensor(args[idx]);

      ICHECK(v.dtype.code == kTVMArgFloat) << "MetaWare Expecting floating point I/O arguments.";

      t.size = 1;
      for (int i = 0; i < v.ndim; i++) t.size *= v.shape[i];

      t.data_type = mwtvm::TvmType::FP32;
      t.data = v.data;
    };

    const mwtvm::GraphInformation& gi = graph_info_;

    ICHECK(args.size() == gi.num_inputs_ + gi.num_outputs_)
        << "Inconsistent arg count between TVM and MWTVM";

    mwtvm::IOTensor sg_inputs[gi.num_inputs_];

    mwtvm::IOTensor sg_outputs[gi.num_outputs_];

    for (int i = 0; i < gi.num_inputs_; i++) {
      SetTensorIO(i, sg_inputs[i]);
    }

    for (int o = 0; o < gi.num_outputs_; o++) {
      SetTensorIO(o + gi.num_inputs_, sg_outputs[o]);
    }

    std::string msg;

    ICHECK(mwtvm::Execute(graph_descriptor_, sg_inputs, sg_outputs, msg)) << msg;
  }

  /* Override GetFunction to reimplement Run method */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& ptr_to_self) override {
    if (this->symbol_name_ == name) {
      return PackedFunc([ptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";

        Run(args);
      });
    } else if (name == "get_symbol") {
      return PackedFunc(
          [ptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_name_; });
    } else if (name == "get_const_vars") {
      return PackedFunc(
          [ptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->const_names_; });
    } else if ("__init_" + this->symbol_name_ == name) {
      // The function to initialize constant tensors.
      return PackedFunc([ptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.size(), 1U);
        std::lock_guard<std::mutex> guard(this->initialize_mutex_);
        if (!this->initialized_) {
          this->Init(args[0]);
          this->initialized_ = true;
        }
        *rv = 0;
      });
    } else {
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
 * @brief Set the MWTVM runtime working directory
 *
 * This directory is where we can process MWTVM runtime artifacts.
 */

TVM_REGISTER_GLOBAL("runtime.MetaWareSetWorkDirectory")
    .set_body_typed(mwtvm::SetRuntimeWorkDirectory);

void MetaWareRuntime::Init(const Array<NDArray>& consts) {
  std::string err_message;

  ICHECK(mwtvm::Init(subgraph_number_, compile_subgraph_number_, graph_descriptor_, hash_,
                     mwtvm_binary, err_message))
      << err_message;

  ICHECK(mwtvm::GetInfo(graph_descriptor_, graph_info_, err_message)) << err_message;
}

runtime::Module MetaWareRuntimeCreate(String symbol_name, int subgraph_number, std::string bin_data,
                                      std::string hash, const Array<String>& const_names) {
  auto n = make_object<MetaWareRuntime>(symbol_name, subgraph_number, subgraph_number, bin_data,
                                        hash, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.MetaWareRuntimeCreate").set_body_typed(MetaWareRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metaware_runtime")
    .set_body_typed(MetaWareLoaderBase::LoadFromBinary<MetaWareRuntime>);

}  // namespace tvm::runtime::contrib::metaware
