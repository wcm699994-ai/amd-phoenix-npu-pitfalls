# AMD Ryzen AI Phoenix(7040 平台)NPU 踩坑全记录

> **English TL;DR:** On AMD Ryzen 7040 / "Phoenix" NPUs (XDNA1, PCI `1022:1502`), **transformer / LLM / OGA / hybrid / new-compiler (Flex-VAIML) workloads are all unsupported at the platform level** — hardware context creation fails with `0xc01e0009` (confirmed by [amd/RyzenAI-SW#365](https://github.com/amd/RyzenAI-SW/issues/365) and AMD's own compatibility matrix: Phoenix/Hawk Point = "CNN INT8" only). Hand-assembled pip environments **cannot drive the DPU at all** (even AMD's own quicktest ResNet gets 0 NPU operators — only the installer-created environment ever worked). This doc catalogs every pitfall we hit while trying to run real-time speech recognition + translation on this NPU, the working recipes we found, and the CPU fallback that actually shipped.

**适用范围:** Ryzen AI Software 1.8.0 / NPU 驱动 32.0.20101.3760 / Windows 11 26200 / Python 3.12 / 2026-10。后续版本可能改善,请以官方文档为准。

---

## 一、背景与结论

**机器:** 七彩虹 TX Gaming FA608UM,AMD Ryzen 7040 平台(Phoenix),NPU = XDNA1 "NPU Phoenix"(PCI `1022:1502`,10~16 TOPS),另含 RTX 5060 Laptop 与 Radeon 780M。

**目标:** 用 NPU 跑实时语音识别(Whisper)+ 离线翻译(opus-mt)。

**最终结论(实测 + 官方文档双重确认):**

| 工作负载 | Phoenix/HPT 支持? | 依据 |
|---|---|---|
| INT8 量化 CNN(分类/检测/分割) | ✅ 唯一官方路径 | Ryzen AI 1.8.0 兼容矩阵:PHX/HPT 仅 "CNN INT8" |
| Transformer(Whisper/翻译/BERT 类) | ❌ | 新 Flex/VAIML 编译器上下文创建失败 `0xc01e0009` |
| LLM(OGA / hybrid) | ❌ | [LLM 文档](https://ryzenai.docs.amd.com/en/latest/llm/overview.html):Ryzen AI 7000/8000 的 NPU 列全为 ✗ |
| Windows ML 第三方模型 | ❌ | ExecutionProviderCatalog 中无 VitisAI EP |
| Windows Studio Effects(摄像头特效/语音聚焦) | ✅(OEM 可能不开放) | 微软内置模型走 NPU 驱动 |

关键官方 issue:[amd/RyzenAI-SW#365](https://github.com/amd/RyzenAI-SW/issues/365) —— 多台 Phoenix 机器、多个驱动版本复现 `Failed to create context virtual (0xc01e0009)`,确认是平台级限制而非配置问题。有开发者用开源 [IRON/mlir-aie](https://github.com/Xilinx/mlir-aie) 栈在 Phoenix 上跑通了 Llama,证明**芯片本身有能力**,是软件栈不支持。

---

## 二、踩坑明细

### P1. 新编译器(Flex/VAIML)在 Phoenix 上硬件层失败 ⭐核心坑

- **现象:** 模型编译、算子分区全部正常(报告能给出 NPU 252/505 节点),但首次 `run()` 时:
  ```
  DPU timeout: (Exception type: class xrt::run::command_error),
  Timeout layer name:[subgraph_(...)], Command failed to complete successfully (ERT_CMD_STATE_ERROR)
  ```
  随后表现为 `vaip_core::TarWriter::write` 异常(TarWriter 只是错误上报路径,不是根因)。
- **原因:** Flex 编译器自动选择 `AMD_AIE2P_4x8_CMC_Overlay` 等 Strix 系目标,Phoenix(XDNA1/AIE-ML)硬件拒绝该指令流。
- **排除掉的可能性:** 换 5 种 xclbin、DD_LAYOUT(col1/2/4)、AMDXL_TARGET 环境变量、升级生产版驱动 32.0.20101.3760——全部无效。
- **结论:** 升级驱动、换软件版本(1.23.2 / 1.27 / lemonade 运行时)都救不了,这是平台代际限制。

### P2. pip 环境连传统 X1 流程也推不上 DPU ⭐次核心坑

- **现象:** 拿到安装器的 `vaiml.dll`/`vaip_config.json` 后,用 quicktest 官方配方(`target: X1` + `xlnx_enable_py3_round: 0` + phoenix 4x4 xclbin)跑**官方 quicktest 的 ResNet 模型**,分区报告仍是:
  ```
  CPU 124  VITIS_EP_CPU 276   (0 个 NPU 算子)
  ```
- **对照:** 官方 issue #171 里,1.3.1 时代**安装器创建的环境**在同一模型上报告 `NPU 398, Actually running on NPU 1`。
- **结论:** pip wheel + 手工补 DLL 复刻不出安装器环境的行为。想真正驱动 DPU,必须用官方安装器原样创建环境。
- **附带发现:** 1.27 栈上显式传 xclbin 会直接 `Cannot find or create target with fingerprint=0x080002050018eec1`;1.23.2 栈则静默全 CPU。

### P3. `vaiml.dll` 只存在于安装器,缺它时报错链很迷惑

- pip 装的 `onnxruntime-vitisai` 不带 `vaiml.dll`,VitisAI EP 的 target 注册表为空:
  ```
  Target auto-discovery: target proto not found for target: X2, valid target names: "VAIML"
  Exception during initialization: not a valid target name, valid names:"VAIML"
  ```
  (不传 config 时 valid names 直接为空)
- 显式 `target: VAIML` 后变成:`Cannot load vaiml.dll: 找不到指定的模块`,然后静默回退全 CPU。
- **获取方式(不运行安装器):**
  ```bash
  7z x ryzen-ai-1.8.0.exe -o<dir>          # 得到 msi + cab
  msiexec /a ryzen-ai-1.8.0.msi /qn TARGETDIR=C:\extracted   # 管理员安装模式 = 纯解包
  # C:\extracted\PFiles64\RyzenAI\1.8.0\
  #   voe-4.0-win_amd64\vaip_config.json
  #   onnxruntime\bin\vaiml.dll, FlexMlMLIRCAPI.dll, z.dll
  ```
  注意:**安装器里没有 xclbin**(`voe-4.0-win_amd64\xclbins` 为空),xclbin 用驱动自带的 `C:\Windows\System32\AMD\4x4_3.5.0.0-2352.xclbin`(fingerprint 与 config_xclbin.txt 一致)。
- 另一条公开渠道:[lemonade-sdk/ryzenai-server](https://github.com/lemonade-sdk/ryzenai-server/releases) 的 zip(569MB,免 EULA),含 `onnxruntime_providers_vitisai.dll` 全套,但同样面向 Strix。

### P4. numpy 必须 <2,scipy 版本会打架

- `onnxruntime-vitisai` 的 C 扩展按 NumPy 1.x 编译,numpy 2.x 下直接 `ImportError: import numpy failed`。
- 但 `pip install --force-reinstall scipy` 会顺带把 numpy 顶到 2.x;且实测出现过 scipy 1.18 dist-info 与 1.9 时代文件混合的损坏安装(scipy/sparse/_sputils.py 引用已删除的 `np.long`)。
- **稳态组合:** `numpy==1.26.4 + scipy==1.13.1`。

### P5. X1 传统后端的量化硬约束(逐条实测)

1. **uint8 激活零点必须 == 128**,否则该算子被踢出 NPU:
   ```
   cancel xir conversion, uint8 data_type only support zero_point is 128.
   this zero_point is: 73
   ```
   ORT 量化器的 `ActivationSymmetric=True` 给的是 127,需要后处理改写(见 [recipes/fix_zero_point_128.py](recipes/fix_zero_point_128.py),实测改写 68 处)。1 LSB 的偏置对转写质量影响可忽略。
2. **Conv 仅支持 2D kernel**,1D 卷积被踢:`'kernel_shape' only support 2-dims convert / cancel xir conversion to conv1d`。Whisper 编码器入口的两个 Conv1d 需改写为 Conv2d(weight unsqueeze 到 4D、`pads/strides/dilations` 从 1 维扩到 2 维、`pads` 从 2 值扩到 4 值),见 [recipes/conv1d_to_conv2d.py](recipes/conv1d_to_conv2d.py)。改写时的四个附加坑:
   - protobuf 7.x **禁止 repeated 字段项赋值**(`conv.input[1] = x`、`g.node[:] = ...` 都报 `does not support assignment`),要用 `insert + del`;
   - INT 类型的 `group` 属性数据在 `i` 字段,往 `ints` 里写会报 `type field and data field mismatch`,要整个 `CopyFrom(helper.make_attribute("group", 1))`;
   - 遍历 attribute 时同时 append 会污染迭代(protobuf),先 `list()` 快照;
   - 改完图必须 `del g.value_info[:]`,否则 opset 转换留下的过期 3D 形状注记会让 ORT 报 `MergeShapeInfo` / `Mul Incompatible dimensions`。
3. **动态形状直接 fatal:**
   ```
   Check failed: input_shape[i] != -1 (-1 vs. -1)
   don't support other shape dimension except batchis -1 now
   ```
   所有输入必须固化静态形状(opset 11 的 Xenova 模型建议先 `onnx.version_converter` 升到 17,Unsqueeze/Squeeze 的 axes 才是输入张量而非属性)。
4. 逐通道量化(PerChannel)在 AMD ORT 上报 `Unrecognized attribute: axis for operator DequantizeLinear`,只能 per-tensor。

### P6. 量化质量:U8S8 完胜 S8S8,校准数据必须是真实的

| 方案 | whisper-tiny 转写质量(jfk.flac) |
|---|---|
| FP32 | `And so my fellow Americans ask not what your country can do for you ask what you can do for your country.`(完美) |
| S8S8 对称 + 随机数据校准 | 输出为空 |
| S8S8 + 熵校准 | 输出为空 |
| **U8S8(QUInt8+对称)+ 真实语音校准** | 与 FP32 仅 1 词之差 |

翻译同样:**用随机 token 做校准时,opus-mt 输出 `282. 277. 277. ...` 乱码**;换成真实句子的 token + pad 掩码校准后,同一模型输出 `天气不错,我们去玩吧`。校准数据的分布必须贴近真实输入。

### P7. optimum/Xenova 合并解码器的三个坑

1. `use_cache_branch` 输入是 **rank-1 bool 张量**:`np.array([False])`。传 python `bool` 或 `np.bool_` 分别报 `'bool' object is not iterable` / `Unable to handle object of type <class 'numpy.bool_'>`;传 0 维 `np.asarray(False)` 报 `Invalid rank Got: 0 Expected: 1`。
2. KV 缓存名映射是 **`present.N.xxx` → `past_key_values.N.xxx`**(不是 `past.N.xxx`,写错报 `Invalid input name: past.5.decoder.key`)。
3. 首步的 past 张量:decoder 自注意力的 KV 传 **长度 0** 的零张量 `[1, H, 0, D]`,cross-attention 的传 `[1, H, enc_len, D]`,配合 `use_cache_branch=False` 可正常走 If 分支。

### P8. 单进程只允许一个 VitisAI 会话

官方已确认:同进程内多个 VitisAI 会话只有第一个正常([相关 issue](https://github.com/amd/RyzenAI-SW/issues))。多模型管线要把每个 NPU 会话放进独立子进程(`multiprocessing` spawn)。

### P9. Windows ML 路线(免 EULA 装 EP)在你机器上不通

`wasdk-Microsoft.Windows.AI.MachineLearning==2.3.0` 装好后,`ExecutionProviderCatalog` 只列出 `MIGraphX / NvTensorRTRTX / WebGpu`,**没有 VitisAI**。另外 python 投影里 `InitializeOptions` 没有 `ON_NO_MATCH_DO_NOT_SHOW_UI`(实际成员是 `ON_NO_MATCH_SHOW_UI` 等),`ensure_ready_async` 会因缺少 Windows App SDK 运行时静默卡死。

### P10. 中文 Windows 的解码坑

官方 `quicktest.py` 用 `pnputil` 枚举设备后按 UTF-8 解码,中文系统输出 GBK 直接 `UnicodeDecodeError (0xb9)`。复现脚本记得 `errors="ignore"` 或自己写探测。

### P11. 网络与分发渠道

- AMD wheel 源:`https://pypi.amd.com/simple`(主)+ `https://pypi.amd.com/ryzenai_llm/<版本>/windows/simple`(LLM 配套,含 382MB 的 `dyn_bins.dll` 所在的 `onnxruntime-providers-ryzenai`)。
- `voe` 老版本(≤1.4,含传统运行时)不在任何公开源上,只在历史安装器 zip 里。
- NPU 驱动有免登录直链:`https://download.amd.com/opendownload/RyzenAI/Driver/NPU_RAI_376_WHQL.zip`;而 Ryzen AI 软件安装器走 EULA 页(account.amd.com,程序化访问会连接失败)。
- GitHub 上的 [lemonade-sdk/ryzenai-server](https://github.com/lemonade-sdk/ryzenai-server/releases) 是免 EULA 的完整运行时 zip(但仍面向 Strix)。

---

## 三、尝试矩阵(全部实测)

| # | 尝试 | 结果 |
|---|---|---|
| 1 | xrt-smi validate | ✅ PASSED(延迟 95µs,吞吐 33544 ops)——硬件没问题 |
| 2 | pip onnxruntime-vitisai 1.23.2,自动目标 | 编译分区 OK(NPU 252/505)→ 运行时 `0xc01e0009` |
| 3 | 同上 + 显式 5 种 xclbin | `Cannot find or create target with fingerprint=...`(旧栈)/ 全 CPU |
| 4 | 1.27 LLM 栈(pip,含 dyn_bins.dll 382MB) | 同 2 |
| 5 | lemonade ryzenai-server 运行时(569MB 免 EULA 包) | 同 2 |
| 6 | DD_LAYOUT / AMDXL_TARGET / 生产版驱动升级 | 无效 |
| 7 | Windows ML EP 目录 | 无 VitisAI EP |
| 8 | X1 传统流程 + 安装器 vaiml.dll + quicktest 配方 | 会话创建成功、能跑,但 **0 NPU 算子**(resnet 与 whisper 均如此) |
| 9 | whisper-tiny FP32(CPU) | ✅ 转写完美 |
| 10 | whisper-tiny U8S8(CPU) | ✅ 仅 1 词之差 |
| 11 | opus-mt en↔zh U8S8(CPU) | ✅ 翻译正确 |
| 12 | 安装器环境(1.3.1,#171 他人复现) | ✅ resnet NPU 398 算子(仅 CNN;1.8.0 未验证成功) |

---

## 四、给 Phoenix / Hawk Point 用户的建议

1. **不要重复我们踩过的坑**:LLM/ASR/翻译/任何 transformer 上 NPU,在当前官方栈上不可行,平台级限制。
2. **想用 NPU**:只走官方安装器流程 + Model Zoo 的 CNN 示例(ResNet/YOLO/超分),并对"新示例只验证 AI 300"有预期。
3. **做实际应用**:CPU 完全够用——我们最终交付的实时字幕+翻译就是纯 CPU(U8S8 INT8 量化)方案,whisper-tiny 转写近乎无损、opus-mt 翻译正确、实时性充裕;降噪用 RNNoise/DeepFilterNet;有 N 卡的话 NVIDIA Broadcast / RTX VSR 是白捡的。
4. **想要完整 NPU 生态(LLM 等)**:升级 AI 300 系列硬件,或者用 GPU。

## 五、参考链接

- [Ryzen AI 软件文档](https://ryzenai.docs.amd.com/en/latest/) · [兼容矩阵(relnotes)](https://ryzenai.docs.amd.com/en/latest/relnotes.html) · [NPU 部署/图切分](https://ryzenai.docs.amd.com/en/latest/modelrun.html)
- [官方 issue #365:0xc01e0009 平台级确认](https://github.com/amd/RyzenAI-SW/issues/365) · [#171:1.3.1 quicktest 在 Phoenix 成功的历史对照](https://github.com/amd/RyzenAI-SW/issues/171)
- [RyzenAI-SW 仓库](https://github.com/amd/RyzenAI-SW)(含 CNN 示例/超分)· [onnxruntime VitisAI EP 文档](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- wheel 源:[pypi.amd.com](https://pypi.amd.com/simple) · [ryzenai_llm 索引](https://pypi.amd.com/ryzenai_llm/1.8.0/windows/simple)
- 驱动直链:`https://download.amd.com/opendownload/RyzenAI/Driver/NPU_RAI_376_WHQL.zip`

## recipes/ 目录

| 文件 | 用途 |
|---|---|
| [`x1_provider_options.py`](recipes/x1_provider_options.py) | Phoenix X1 传统路径的完整会话配置(quicktest 官方配方 + 环境变量) |
| [`fix_zero_point_128.py`](recipes/fix_zero_point_128.py) | X1 要求的 uint8 零点=128 后处理 |
| [`conv1d_to_conv2d.py`](recipes/conv1d_to_conv2d.py) | Conv1d → Conv2d 图改写(X1 只支持 2D kernel),含 protobuf 7 兼容写法 |
