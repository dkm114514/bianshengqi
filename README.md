# 实时变声器

用麦克风实时变声。选择麦克风和输出设备，就可以开始使用；变声、降噪、声纹和耳机监听都在同一个界面里。

## 下载和安装

前往 [GitHub Releases](https://github.com/dkm114514/bianshengqi/releases/latest)，下载这三个文件，并放在同一个文件夹：

- bianshengqi-0.1.0-windows-x64-setup.exe
- bianshengqi-0.1.0-windows-x64-setup-1.bin
- bianshengqi-0.1.0-windows-x64-setup-2.bin

双击 EXE，按提示完成安装。这三个文件合在一起就是完整安装包；Python、RVC 和所需模型都已经装在里面，安装和首次启动都不需要联网下载。

## 第一次使用

安装时可以选择安装包内附带的 VB-CABLE 虚拟声卡。电脑里已经有 VB-CABLE 的话，可以跳过。安装驱动后按安装程序提示重启电脑。

启动变声器后，选择自己的麦克风和耳机。如果想把变声后的声音送进游戏或聊天软件，在变声器里选择 CABLE Input 作为输出，再在游戏或聊天软件里选择 CABLE Output 作为麦克风。

默认音色和常用参数已经设置好。声纹需要自己录入，或从之前导出的文件导入。声纹可以随时导出；删除或更换前程序会先自动备份。安装包不包含个人声纹。

## 运行环境

这是 Windows x64、NVIDIA CUDA 版本，目前已在 RTX 4060 上验证。其他显卡和系统尚未验证。

第三方组件和 VB-CABLE 的说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
