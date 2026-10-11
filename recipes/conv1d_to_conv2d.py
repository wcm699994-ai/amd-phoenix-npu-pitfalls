"""把 Conv1d(rank-3 权重)改写为 Conv2d —— X1 传统后端仅支持 2D kernel。

whisper 编码器示例:input [1,80,3000] --conv1(4D)--> [1,384,1,3000]
--GELU(逐元素,秩无关)--> --conv2(4D)--> [1,384,1,3000] --Squeeze--> [1,384,3000]

踩坑点(全部实测,详见 README P5/P11):
- protobuf 7 禁止 repeated 字段项赋值 → insert+del;
- INT 属性(group)数据在 `i` 字段,改 ints 会类型/数据不匹配 → 整体 CopyFrom;
- 遍历 attribute 时 append 会污染迭代 → list() 快照;
- opset <13 时 Unsqueeze/Squeeze 的 axes 是属性,建议先把模型升到 opset 17
  (onnx.version_converter),否则 AMD ORT 的形状推断会出现诡异报错;
- 改完必须删光 value_info,否则旧 3D 形状注记导致 Mul/Reshape 维度合并失败;
- Squeeze 节点必须紧跟最后一个 conv,放图末尾会破坏拓扑序。

用法: python conv1d_to_conv2d.py model_static_op17.onnx model_2d.onnx
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _set(vec, idx, val):
    vec.insert(idx, val)
    del vec[idx + 1]


def transform(model_path: str, out_path: str):
    m = onnx.load(model_path)
    g = m.graph

    old_input = g.input[0]
    old_name = old_input.name
    dims = [d.dim_value for d in old_input.type.tensor_type.shape.dim]
    assert dims == [1, 80, 3000], f"预期 whisper 形状,实际 {dims}——其他模型请调整"

    inits = {t.name: t for t in g.initializer}
    convs = [n for n in g.node if n.op_type == "Conv" and len(inits[n.input[1]].dims) == 3]
    assert convs, "未找到 rank-3 Conv"

    new_inits = []
    before = {}   # conv 名 -> 前置 Unsqueeze 节点
    producers = {o: n for n in g.node for o in n.output}

    # 1) 图输入改为 4D
    new_inp = helper.make_tensor_value_info(old_name + "_4d", TensorProto.FLOAT,
                                            [1, 1, 80, 3000])

    def mk_unsqueeze(src, dst, name, axes):
        ax = numpy_helper.from_array(np.array(axes, dtype=np.int64), name + "_axes")
        new_inits.append(ax)
        return helper.make_node("Unsqueeze", [src, name + "_axes"], [dst], name=name)

    def mk_squeeze(src, dst, name, axes):
        ax = numpy_helper.from_array(np.array(axes, dtype=np.int64), name + "_axes")
        new_inits.append(ax)
        return helper.make_node("Squeeze", [src, name + "_axes"], [dst], name=name)

    for idx, conv in enumerate(convs):
        # 权重 3D -> 4D
        w = numpy_helper.to_array(inits[conv.input[1]])
        w4 = w.reshape(w.shape[0], 1, *w.shape[1:])
        new_inits.append(numpy_helper.from_array(w4, conv.input[1] + "_4d"))
        _set(conv.input, 1, conv.input[1] + "_4d")

        # group 修正(INT 属性整体替换,避免类型/数据不匹配)
        group_fixed = False
        for ai, a in enumerate(list(conv.attribute)):
            if a.name == "group":
                conv.attribute[ai].CopyFrom(helper.make_attribute("group", 1))
                group_fixed = True
        if not group_fixed:
            conv.attribute.append(helper.make_attribute("group", 1))

        # 空间属性 1D -> 2D
        for a in list(conv.attribute):
            if a.name == "kernel_shape" and len(a.ints) == 1:
                v = a.ints[0]; del a.ints[:]; a.ints.extend([1, v])
            elif a.name in ("strides", "dilations") and len(a.ints) == 1:
                v = a.ints[0]; del a.ints[:]; a.ints.extend([1, v])
            elif a.name == "pads" and len(a.ints) == 2:
                b, e = a.ints[0], a.ints[1]; del a.ints[:]
                a.ints.extend([0, b, 0, e])

        # 输入补 Unsqueeze(上游是 conv 时输出已是 4D,无需)
        src = conv.input[0]
        if src == old_name:
            _set(conv.input, 0, old_name + "_4d")
        else:
            producer = producers.get(src)
            if not (producer is not None and producer.op_type == "Conv"):
                before[conv.name] = mk_unsqueeze(src, src + "_4d",
                                                 f"unsqueeze_{idx}", [2])
                _set(conv.input, 0, src + "_4d")

    # 2) 最后一个 conv 之后补 Squeeze,恢复 3D 给后续图使用
    last = convs[-1]
    last_out = last.output[0]
    for n in g.node:
        for i, nm in enumerate(n.input):
            if nm == last_out and n is not last:
                _set(n.input, i, last_out + "_4d")
    _set(last.output, 0, last_out + "_4d")
    squeeze_node = mk_squeeze(last_out + "_4d", last_out, "squeeze_output", [2])

    # 3) 按拓扑序重组(Squeeze 紧跟 last conv)
    out_nodes = []
    for n in g.node:
        if n.name in before:
            out_nodes.append(before[n.name])
        out_nodes.append(n)
        if n is last:
            out_nodes.append(squeeze_node)

    keep_inits = [t for t in g.initializer
                  if t.name not in {c.input[1] for c in convs}] + new_inits
    del g.node[:]
    g.node.extend(out_nodes)
    del g.initializer[:]
    g.initializer.extend(keep_inits)
    g.input.remove(old_input)
    g.input.append(new_inp)
    del g.value_info[:]  # 旧 3D 形状注记必须清除

    onnx.checker.check_model(m)
    onnx.save(m, out_path)
    print(f"wrote {out_path}(输入: {old_name}_4d [1,1,80,3000])")


if __name__ == "__main__":
    transform(sys.argv[1], sys.argv[2])
