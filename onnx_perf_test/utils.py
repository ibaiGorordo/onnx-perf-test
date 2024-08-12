import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

provider_map = {
    "TENSORRT": ["TensorrtExecutionProvider"],
    "CUDA": ["CUDAExecutionProvider"],
    "MIGRAPHX": ["MIGraphXExecutionProvider"],
    "ROCM": ["ROCMExecutionProvider"],
    "OPENVINO": ["OpenVINOExecutionProvider"],
    "DNNL": ["DnnlExecutionProvider"],
    "TVM": ["TvmExecutionProvider"],
    "VITISAI": ["VitisAIExecutionProvider"],
    "QNN": ["QNNExecutionProvider"],
    "VSINPU": ["VSINPUExecutionProvider"],
    "JS": ["JsExecutionProvider"],
    "COREML": ["CoreMLExecutionProvider"],
    "ARMNN": ["ArmNNExecutionProvider"],
    "ACL": ["ACLExecutionProvider"],
    "DML": ["DmlExecutionProvider"],
    "RKNPU": ["RknpuExecutionProvider"],
    "WEBNN": ["WebNNExecutionProvider"],
    "XNNPACK": ["XnnpackExecutionProvider"],
    "CANN": ["CANNExecutionProvider"],
    "AZURE": ["AzureExecutionProvider"],
    "CPU": ["CPUExecutionProvider"],
    "DEFAULT": onnxruntime.get_available_providers()
}

tensor_type_map = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(int16)": np.int16,
    "tensor(uint16)": np.uint16,
    "tensor(int64)": np.int64,
    "tensor(uint64)": np.uint64,
}


def tensor_dtype_to_np_dtype(tensor_dtype: str) -> np.dtype:
    """ Convert a tensor data type to a numpy data type """
    return tensor_type_map[tensor_dtype]


def create_test_inputs(session: onnxruntime.InferenceSession) -> dict[str, np.ndarray]:
    input_info = session.get_inputs()
    inputs = {}
    for input in input_info:
        inputs[input.name] = np.random.rand(*input.shape).astype(tensor_dtype_to_np_dtype(input.type))

    return inputs


def tensors_to_string(tensors: list[onnxruntime.NodeArg]) -> str:
    """ Convert a list of NodeArg objects to a string """
    return "\n" + "\n".join([f"\t{input.name} - Shape: {input.shape}, Type: {input.type}" for input in tensors])


def draw_bar_stats(stats: pd.DataFrame, ax: plt.Axes, max_values=10):
    if len(stats)-2 < max_values:
        max_values = len(stats)-2

    selected_stats = stats[:max_values+2]
    colors = [np.ones(3) * 0.3, np.ones(3) * 0.6] + [plt.colormaps['Spectral'](i) for i in np.linspace(0, 1, max_values)]
    selected_stats["mean"].plot(kind="barh", xerr=selected_stats["std"], ax=ax, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Mean duration (ms)")
    ax.set_title(f"Top {max_values} nodes")


def draw_pie_stats(stats: pd.DataFrame, ax: plt.Axes, max_values=10):
    if len(stats)-2 < max_values:
        max_values = len(stats)-2

    selected_stats = stats[2:max_values+2]
    other_stats = stats[max_values+2:]
    other_stats = pd.DataFrame({"mean": other_stats["mean"].sum(), "std": other_stats["std"].sum()}, index=["Other"])
    selected_stats = pd.concat([selected_stats, other_stats])

    colors = [plt.colormaps['Spectral'](i) for i in np.linspace(0, 1, max_values)] + [np.ones(3) * 0.8]
    selected_stats["mean"].plot(kind="pie", ax=ax, colors=colors, autopct="%1.1f%%", startangle=0, counterclock=True)
    ax.set_ylabel("")


def draw_results(stats: pd.DataFrame, max_values=10):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    draw_bar_stats(stats, axs[0], max_values)
    draw_pie_stats(stats, axs[1], max_values)
    plt.show()
