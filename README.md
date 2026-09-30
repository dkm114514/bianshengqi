# 实时变声器

面向 Windows 的 RVC 实时变声桌面程序，使用 FreeSimpleGUI 和 WASAPI。
支持模型切换、音高与共振峰调整、降噪、声纹过滤、自动增益、VB-CABLE 输出和耳机监听。

音频链路：麦克风 → 降噪 / 声纹过滤 → RVC → SOLA 拼接 → 自动增益 → 虚拟麦克风 / 监听。

本仓库提供源码。RVC 整合包、模型权重、音频驱动和个人声纹需要自行准备。
目前验证环境为 Windows、NVIDIA CUDA、RVC 内置 Python 3.9，未验证其他平台和新版本 RVC。

## 准备环境

1. 准备可运行的 [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) Windows 整合包。
   本程序调用其中的 `infer/lib/rtrvc.py` 和 `tools/torchgate`。
   保留整合包原有的 PyTorch、torchaudio、CUDA 和 fairseq 配套版本。
2. 使用该整合包的 Python 安装附加依赖，而不是系统 Python：

   ```powershell
   $env:RVC_ROOT = 'D:\RVC'
   & "$env:RVC_ROOT\runtime\python.exe" -m pip install -r requirements.txt
   ```

   `requirements.txt` 记录本机验证过的附加依赖版本；它不是一个全新的 RVC 环境安装清单。
   对应已验证核心版本为 torch 2.0.0+cu118、torchaudio 2.0.1。
3. 安装 [VB-CABLE](https://download.vb-audio.com/) 后，在聊天或录音软件中选择它的录音端点（通常是 `CABLE Output`）。
   本程序输出端选择 `CABLE Input`；重命名后的端点也可用。

## 配置运行路径

任选一种方式：

- 设置环境变量 `BSQ_RVC_ROOT` 或 `RVC_ROOT`，值为包含 `runtime/`、`infer/`、`assets/` 的 RVC 根目录。
- 设置环境变量 `RVC_RUNTIME`，值为 RVC 的 `runtime/python.exe` 完整路径。
- 在项目根目录创建 `runtime.local.txt`，写入该 Python 的完整路径，仅一行，不加引号。
- 将整合包放到项目根目录的 `RVC/` 中。

优先级为 `BSQ_RVC_ROOT` → `RVC_ROOT` → `RVC_RUNTIME` → `runtime.local.txt` → `RVC/`。
请勿同时设置指向不同安装的根目录和 Python 路径。`runtime.local.txt` 不上传。

双击 `启动变声器.bat`，或从 PowerShell 启动：

```powershell
$env:RVC_ROOT = 'D:\RVC'
& "$env:RVC_ROOT\runtime\python.exe" -X utf8 main.py
```

模型权重默认读取 RVC 的 `assets/weights/*.pth`，索引读取 `logs/*.index`。
也可放在项目自身的 `assets/weights/` 和 `logs/`，或通过 `BSQ_WEIGHTS_DIR`、`BSQ_INDEX_DIR` 指定。
索引按权重同名匹配；`bb48k.pth` 特别匹配 `guanguanV1.index`。没有索引也可使用模型。
模型文件均不随源码发布。

## 声纹和降噪模型

启用声纹过滤前，需在项目 `models/` 放置兼容的 CAM++ 和 Silero ONNX。
已验证文件为：

- `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`
- `silero_vad.onnx` 或 `silero_vad_v5.onnx`

声纹模型可从 [sherpa-onnx 官方模型列表](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html) 获取。
声纹迁移时应使用相同模型，192 维声纹不能直接用于其他维度的模型。

DFN3 首次运行会下载模型到 `models/dfn3-512-v1/` 并校验哈希；离线使用需提前准备其中的
`denoiser_model.onnx`、`initial_states.npz`、`meta.json`。
下载来源、镜像和校验逻辑见 `dsp/dfn.py`。首次使用可先关闭未准备好的声纹过滤。

## 使用

选择变声模型、麦克风、监听设备后启动。修改音高、共振峰、音量跟随和降噪开关可实时生效；
块长、额外推理时长、交叉淡化和 CPU 线程数需要重建音频链路。

“**一起刷新**”会重新枚举并重开麦克风、虚拟输出和监听，清空音频、音高、声纹判定、降噪及增益状态，
保留已加载模型。运行中刷新有短暂停顿；设备打开失败会尝试恢复原设备。

声纹支持录入、验证、删除、导入和导出：

- 当前声纹为 `models/voice_profile.npz`。
- 导出 `.npz` 保留原文件字节；`.json` 可用于迁移检查。
- 导入先验证文件，再备份旧声纹、原子替换；导入后启用声纹过滤。
- 删除或重新录入之前自动备份，位置为 `models/voice_backups/`。
- 个人声纹和备份仅保存在本地，不进入 Git 或源码包。

当前预设参数来自实际使用配置：

| 参数 | 默认值 |
| --- | --- |
| 音高 / 共振峰 | +16 半音 / 0 |
| 索引比例 / 音量跟随 | 0 / 0 |
| 块长 / 交叉淡化 / 额外推理时长 | 0.15 / 0.01 / 4.0 秒 |
| 输入门限 | -60 dB |
| 输入 / 输出降噪 | 开 / 开 |
| 降噪模式 | DFN3 |
| 音高算法 / CPU 线程数 | RMVPE / 4 |
| 声纹过滤 / 相似度阈值 | 开 / 0.2 |
| 自动增益 / 目标电平 | 开 / -20 dBFS |
| 监听 | 开 |

无配置时自动使用内置参数。`profiles.example.json` 是无本机路径的示例，可复制为 `profiles.json` 后调整。
首次使用请在界面选择自己的设备。`profiles.json` 保存各模型和设备的实际设置，不上传。

## 目录

```text
main.py                       启动入口
app/                          界面、声纹文件管理
engine/                       模型扫描、默认参数、RVC 和音频链路
dsp/                          降噪、VAD、声纹、增益、拼接等信号处理
tools/devsetup.py             可选的 Windows 音频端点管理工具
tools/package_source.py       源码打包工具
tests/                        回归测试，使用合成声纹
profiles.example.json         可公开的配置示例
requirements*.txt             附加依赖
```

`tools/devsetup.py` 可枚举、启用端点或设置系统默认录音设备；启用端点需要管理员权限。
需要使用它时，另装 `requirements-devsetup.txt`。`--list` 是只读枚举，其他操作会改变系统设备设置。

## 验证

在已经准备好的 RVC 环境中运行：

```powershell
& "$env:RVC_ROOT\runtime\python.exe" -X utf8 -m unittest discover -s tests -v
& "$env:RVC_ROOT\runtime\python.exe" -X utf8 smoke_test.py --pure
```

回归测试覆盖暂停后起音、异步声纹结果、设备重开和失败恢复、声纹导入导出与备份。
声纹文件集成测试需要上述 CAM++ 和 Silero 模型；模型缺失时会明确跳过。
测试不会读取个人声纹。`--pure` 不打开音频设备，实际驱动兼容性仍需在目标机器验证。
`selfcheck.py` 和其他烟测模式可能打开音频设备，使用前查看 `--help`。

## 打包上传 GitHub

```powershell
& "$env:RVC_ROOT\runtime\python.exe" -X utf8 tools/package_source.py
```

生成 `dist/bianshengqi-source.zip`，包含源码、测试、说明和 SHA-256 文件清单。
打包使用明确的文件允许列表，不包含 `.git`、本机配置、模型、声纹、驱动、输出结果和缓存。

推荐解压源码包，在新目录初始化 Git 仓库或通过 GitHub 网页上传包内文件。
不要把 ZIP 本身当作仓库源码上传。
旧本地 Git 历史中曾包含驱动包和本机配置；`.gitignore` 和移出跟踪不会清除历史。
直接推送旧仓库会带上这些历史内容；本源码包不携带历史。

暂未添加源码许可证。RVC、音频驱动、第三方依赖及模型各自的许可仍需遵守。
