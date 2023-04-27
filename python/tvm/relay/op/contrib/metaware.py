# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""METAWARE library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by METAWARE.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to METAWARE.
"""
import logging
from functools import reduce

import tvm.ir
from tvm import relay
from tvm.ir import Op
from tvm.relay import expr as _expr
from tvm.relay import transform
from tvm.relay.analysis import analysis as _analysis
from tvm.relay.expr import Call, GlobalVar, TupleGetItem, const
from tvm.relay.expr_functor import ExprMutator, ExprVisitor
from tvm.contrib.target.onnx import to_onnx

from tvm.relay.build_module import bind_params_by_name

from ... import _ffi_api
from ...dataflow_pattern import (
    DFPatternCallback,
    is_constant,
    is_expr,
    is_op,
    rewrite,
    wildcard,
)
from .register import register_pattern_table

logger = logging.getLogger("METAWARE")


def get_attrs(expr):
    """Get the attributes from an expression."""
    if isinstance(expr, Call):
        return expr.attrs
    if isinstance(expr, TupleGetItem):
        return get_attrs(expr.tuple_value)
    return {}


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by METAWARE.

    Parameters
    ----------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by METAWARE.
    """

    @tvm.ir.register_op_attr(op_name, "target.metaware")
    def _func_wrapper(expr):
        args = expr.args
        if any([x.checked_type.dtype == "int64" for x in args]):
            logger.info("METAWARE does not support int64.")
            return False
        # check: does METAWARE support pooling with ceil_mode = True?
        if "pool" in op_name:
            attrs = dict(get_attrs(expr))
            if "ceil_mode" in attrs.keys() and attrs["ceil_mode"]:
                return False

        return supported

    return _func_wrapper


'''
Start with simple operators for testing!
'''

_register_external_op_helper("add")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.maxpool_2d")


def partition_for_metaware(mod, params=None, mod_name="default"):
    """Partition the graph greedily offloading supported
    operators to Synopsys MetaWare compiler.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.AnnotateTarget("metaware"),
            transform.MergeCompilerRegions(), # some targets do this after partition?
            transform.PartitionGraph(),
            transform.InferType()
        ]
    )

    return seq(mod)
