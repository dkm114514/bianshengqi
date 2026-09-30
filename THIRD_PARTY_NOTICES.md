# 第三方组件说明

离线发行包包含 RVC 引擎、Python、PyTorch/CUDA、ONNX Runtime、FreeSimpleGUI、
音频和科学计算依赖。各组件的原许可及版权文件保留在 `RVC/` 和
`RVC/runtime/Lib/site-packages/*dist-info/` 等目录中。

- RVC：https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
  原 MIT 许可证及引用库说明保留在 `RVC/LICENSE` 与 `RVC/THIRD_PARTY_LICENSES.txt`。
- CAM++ 与 Silero：https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
- DeepFilterNet3 流式实现：https://github.com/wuxuedaifu/deepfilter-stream
- Inno Setup（安装包制作工具）：https://jrsoftware.org/isinfo.php

离线包附带原版 VB-CABLE 安装文件。VB-CABLE 来自 VB-Audio：
https://vb-audio.com/Cable/ （https://www.vb-cable.com/）。
它是 **donationware**；如果觉得有用，欢迎到厂商网站捐赠或支付许可证费用。
它由独立的原版安装程序安装，用户可阅读并决定接受其许可。
发行条件：https://vb-audio.com/Services/licensing.htm 。
只随包附带普通 VB-CABLE，不包含 A+B、C+D 或 Voicemeeter Potato。

默认音色为当前使用的 `bb48k.pth`，本次发行存放在私有仓库。
个人声纹、声纹备份、个人配置和训练素材均不随包分发。
本项目源码暂未另行选择许可证；上述第三方组件仍按各自许可使用。
