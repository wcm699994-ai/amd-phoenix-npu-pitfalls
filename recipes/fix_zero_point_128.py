"""X1 传统后端要求 uint8 激活的零点(zero_point)恰好为 128。

ORT 量化器在 ActivationSymmetric=True 时对 QUInt8 给出的 zp 是 127,
不符合 X1 要求,会被逐算子踢出 NPU:
    cancel xir conversion, uint8 data_type only support zero_point is 128.
本脚本把图中所有 uint8 零点改写为 128(引入 1 LSB 的均匀偏置,
实测对 whisper-tiny 转写质量无可感知影响)。

用法: python fix_zero_point_128.py model_int8.onnx
"""
import sys

import onnx
import numpy as np
from onnx import numpy_helper


def main(path: str):
    m = onnx.load(path)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    changed = 0
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear") and len(n.input) >= 3:
            t = inits.get(n.input[2])
            if t is not None and t.data_type == onnx.TensorProto.UINT8:
                arr = numpy_helper.to_array(t)
                if arr.size == 1 and arr.flat[0] != 128:
                    arr.fill(128)
                    inits[n.input[2]].CopyFrom(numpy_helper.from_array(arr, n.input[2]))
                    changed += 1
    onnx.save(m, path)
    print(f"rewrote {changed} zero_points to 128 in {path}")


if __name__ == "__main__":
    main(sys.argv[1])
