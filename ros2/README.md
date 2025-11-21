# 2025年笃行战队视觉组代码文档-ROS2版

## 编译相关说明

运行```ros2_compile.sh```即可完成编译。
如果想让clangd插件在ros2的cpp代码中生效的话(这几乎是必须的)，将如下的配置项写入`.vscode/setting.json`中:

```json
"clangd.arguments": [
    "--compile-commands-dir=${workspaceFolder}/ros2/build",
    "--background-index=false"
]
```
即正确的指定```compile_commands.json```的路径即可。

使用include的方式链接原有的文件,仅使用```ros2_main.cpp```一个文件对原有的库(如`armor` `windmill` `params`)进行调用和链接ros2的相关功能,类似于根目录下的`main.cpp`。因此在修改代码的时候可以按照原有的方式对`AimAuto.cpp`等文件进行修改,并同时以原生/ros2驱动两种方式进行编译和运行。