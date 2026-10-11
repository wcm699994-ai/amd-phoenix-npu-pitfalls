"""Phoenix/Hawk Point X1 传统路径的完整会话配置(来自官方 quicktest.py 1.8.0 配方)。

注意:即使完全按此配置,在 pip 手工环境里也只能拿到 0 个 NPU 算子
(必须使用官方安装器创建的环境,见 README P2)。此文件用于:
1) 在安装器环境里部署 CNN INT8 模型;
2) 复现实验/生成算子分配报告。
"""
import os

# ---- 需要加入 PATH / dll 目录的位置 ----
DLL_DIRS = [
    r"C:\Windows\System32\AMD",                        # XRT + vitis-ai-runtime(驱动自带)
    r"C:\Program Files\RyzenAI\1.8.0\onnxruntime\bin", # vaiml.dll / FlexMlMLIRCAPI.dll / z.dll(安装器)
]

# ---- 传统路径环境变量(1.3/1.4 时代 quicktest 配方,issue #171)----
def setup_legacy_env(installation_path: str, xclbin: str):
    os.environ["RYZEN_AI_INSTALLATION_PATH"] = installation_path
    os.environ["XLNX_VART_FIRMWARE"] = xclbin
    os.environ["XLNX_TARGET_NAME"] = "AMD_AIE2_Nx4_Overlay"
    os.environ["NUM_OF_DPU_RUNNERS"] = "1"
    os.environ["XLNX_ENABLE_CACHE"] = "0"


# ---- 官方 quicktest.py(1.8.0)对 PHX/HPT 的 provider 配方 ----
X1_PROVIDER_OPTIONS = {
    "target": "X1",
    "xlnx_enable_py3_round": 0,
    # 驱动自带 xclbin 与安装器 xclbins/phoenix/4x4.xclbin 等价(fingerprint 一致)
    "xclbin": r"C:\Windows\System32\AMD\4x4_3.5.0.0-2352.xclbin",
    "cache_dir": r"C:\temp\vaip_cache",
    "cache_key": "your_model_cache_key",       # 每个模型唯一
    "enable_cache_file_io_in_mem": "0",        # 0 = 启用算子分配报告
    # 可选:安装器的 vaip 配置(影响 pass 链,实测对 DPU 分配无决定性作用)
    # "config_file": r"C:\Program Files\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json",
}


def make_session(model_path: str):
    import onnxruntime as ort

    for d in DLL_DIRS:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
    setup_legacy_env(
        installation_path=r"C:\Program Files\RyzenAI\1.8.0",
        xclbin=X1_PROVIDER_OPTIONS["xclbin"],
    )
    # 算子分配报告(需 enable_cache_file_io_in_mem=0)
    report = os.path.join(X1_PROVIDER_OPTIONS["cache_dir"], "report.txt")
    os.environ["XLNX_ONNX_EP_REPORT"] = report
    os.environ["XLNX_ONNX_EP_REPORT_FILE"] = report

    so = ort.SessionOptions()
    so.log_severity_level = 1  # INFO:能看到 "[Vitis AI EP] No. of Operators : NPU xxx"
    return ort.InferenceSession(
        model_path, sess_options=so,
        providers=["VitisAIExecutionProvider"],
        provider_options=[X1_PROVIDER_OPTIONS],
    )


if __name__ == "__main__":
    import json
    sess = make_session(r"model.onnx")
    print("providers OK; 报告:", report_path := X1_PROVIDER_OPTIONS["cache_dir"] + "\\report.txt")
