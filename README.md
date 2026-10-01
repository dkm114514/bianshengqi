# 实时变声器

用麦克风实时变声，支持 DFN3 降噪、声纹过滤、自动增益和耳机监听。当前版本 **v0.1.1**，改为绿色版。

## 下载和使用

前往 [v0.1.1 下载页面](https://github.com/dkm114514/bianshengqi/releases/tag/v0.1.1)，或直接下载下面三个文件，并放在同一个文件夹。三个都需要下载：

- [bianshengqi-0.1.1-windows-x64-portable.exe](https://github.com/dkm114514/bianshengqi/releases/download/v0.1.1/bianshengqi-0.1.1-windows-x64-portable.exe)
- [bianshengqi-0.1.1-windows-x64-portable-1.bin](https://github.com/dkm114514/bianshengqi/releases/download/v0.1.1/bianshengqi-0.1.1-windows-x64-portable-1.bin)
- [bianshengqi-0.1.1-windows-x64-portable-2.bin](https://github.com/dkm114514/bianshengqi/releases/download/v0.1.1/bianshengqi-0.1.1-windows-x64-portable-2.bin)

双击 EXE，选择一个自己的文件夹并解压，然后运行里面的 **启动变声器.bat**。包里已经带好 Python、RVC 和所需模型，解压和首次使用都不用联网下载。建议放在自己的桌面或其他可写目录里。

## 第一次使用

第一次使用需要 VB-CABLE 虚拟声卡。电脑里已经有的话不用重装；没有的话，运行 **安装或卸载虚拟声卡.bat**，按原版驱动程序的提示安装并重启。

启动变声器后，选择自己的麦克风和耳机，点击启动。在 QQ、游戏或其他聊天软件里，将麦克风选为 **CABLE Output**（或自己重命名过的 VB-CABLE 录音设备），就能使用变声后的声音。

默认音色和常用参数已经设置好。声纹可以自己录入，也可以导入之前保存的文件；支持导入、导出和删除，删除或更换前会自动备份。发行包不包含任何个人声纹。

## 保存、更新和删除

配置、声纹和声纹备份在 `data` 文件夹里，运行日志在 `logs`，缓存和临时文件在 `cache`。移动程序时，把整个文件夹一起移动。

已经有 v0.1.0 离线完整版的用户，可以只下载 [v0.1.1 小更新包（约 2 MB）](https://github.com/dkm114514/bianshengqi/releases/download/v0.1.1/bianshengqi-0.1.1-windows-x64-update.exe)，关闭程序后将更新解压到原来的变声器目录。它只更新程序，不替换模型、运行环境或个人声纹。已有配置和声纹会在首次启动时迁入 `data`。第一次使用请下载完整绿色包。

小更新包不会移除旧安装版的系统安装项。想完全换成绿色版的话，请先导出声纹，再卸载旧版，将完整绿色包解压到另一个文件夹。

绿色版不创建系统安装项或快捷方式。不用了就关闭程序，需要保留声纹的话先导出，再删除整个程序文件夹即可。VB-CABLE 是系统驱动，需要单独卸载：运行随包的驱动程序，选择 **Remove Driver**，完成后重启。其他软件还在用它就保留。

## 这次更新

v0.1.1 把配置、声纹、日志和缓存集中到程序文件夹，修正了保存配置失败后继续报错的问题。同一文件夹只允许启动一个实例，避免多个窗口同时保存配置。麦克风和监听设备会一起记忆，移动整个文件夹后可以继续使用。

## 支持的电脑

这是 Windows x64、NVIDIA CUDA 版本，目前已在 RTX 4060 上验证。其他显卡和系统尚未验证。

第三方组件和 VB-CABLE 的说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
