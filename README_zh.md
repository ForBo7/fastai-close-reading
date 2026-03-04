# fastai-close-reading

![fastai-close-reading](thumbnail.png)

对 Jeremy Howard 主讲的 [《程序员实用深度学习》](https://course.fast.ai) 课程**几乎全部课时**进行结构化精读（更准确地说是「精看」）并整理成文稿——涵盖第 1、2 部分共 28 节课。

不包含 [数据伦理 bonus 课 8a](https://course.fast.ai/Lessons/lesson8a.html)。欢迎你动手把它补充进来！

本仓库主要为在 [SolveIt](https://solve.it.com) 上进行上下文学习而设计，但也可在其他工具中使用。

[English README →](README.md)

## 目录

- [项目介绍](#项目介绍)
- [仓库结构](#仓库结构)
  - [什么是 CRAFT 文件？](#什么是-craft-文件)
  - [summaries/ 文件夹](#summaries-文件夹)
- [使用方法](#使用方法)
  - [在 SolveIt 中使用（推荐）](#在-solveit-中使用推荐)
  - [在 SolveIt 外使用](#在-solveit-外使用)
- [上下文长度注意事项](#上下文长度注意事项)
- [视频帧截取](#视频帧截取)
  - [本地环境依赖](#本地环境依赖)
  - [配置步骤](#配置步骤)
  - [本地运行（不使用 SolveIt）](#本地运行不使用-solveit)
- [已知问题与练习任务](#已知问题与练习任务)
  - [1. Markdown 中的 KaTeX 渲染](#1-markdown-中的-katex-渲染)
  - [2. 标题粒度](#2-标题粒度)
  - [3. 缺失数据伦理课程](#3-缺失数据伦理课程)
  - [4. 章节粒度](#4-章节粒度)
- [相关链接](#相关链接)
- [致谢](#致谢)

---

## 项目介绍

本仓库包含 fast.ai《程序员实用深度学习》课程**几乎所有课时**的结构化文稿，以 Jupyter Notebook 形式呈现。每节课都被拆分为多个片段，包含：

- Jeremy 与课程团队的讲解与演示**完整内容**
- 通过 `fetch_frame()` 调用获取的**视频帧引用**（幻灯片、图表、代码演示等）
- **上下文关联**——自动加载上一节、当前节、下一节课的总结（在 SolveIt 中会自动载入）
- 按课时与课程部分整理的**挑战、作业、研究方向与资源**

目标是**从课程内容中「读出来」**，而不是被动「读进去」。帮助你以更具批判性的方式学习，更深入地理解知识点，并保持专注流畅的学习状态。关于精读的更多说明见[这里](https://www.fast.ai/posts/2026-01-21-reading-LLMs/)。

## 仓库结构

```
.
├── CRAFT.ipynb                     # 课程总览总结
├── CONTROLLER.ipynb                # 管理面板（课时目录）
├── part1/
│   ├── CRAFT.ipynb                 # 第 1 部分总结、挑战、资源
│   ├── lesson0/ → lesson8/
│   │   └── CRAFT.ipynb             # 单课时：总结 + 精读笔记
├── part2/
│   ├── CRAFT.ipynb                 # 第 2 部分总结、挑战、资源
│   ├── lesson9/ → lesson25/
│   │   └── CRAFT.ipynb             # 单课时：总结 + 精读笔记
└── summaries/
    ├── all.md                      # 课程完整汇总
    ├── part1.md / part2.md         # 分部分总结
    └── lesson0.md ... lesson25.md  # 单课时独立总结
```

### 什么是 CRAFT 文件？

在 [SolveIt](https://solve.it.com) 中，**CRAFT.ipynb** 为对应文件夹提供大模型上下文。当你在某个课时文件夹内打开对话时，SolveIt 会自动加载该文件夹及所有上层文件夹的 CRAFT 文件，形成**层级式上下文**：

| 层级 | 包含内容 |
|------|----------|
| **根目录** | 完整课程总览 |
| **课程部分**（part1/、part2/） | 对应部分总结、挑战、作业、研究方向、资源 |
| **单课时**（如 part1/lesson3/） | 上一节 + 当前节 + 下一节课总结、视频 ID、带帧截取的逐段精读笔记 |

打开 `part1/lesson3/` 内的对话，你会在输入第一条提示前，就自动获得：课程总览、第 1 部分总结、第 3 节课完整精读笔记。

### summaries/ 文件夹

每节课内容叙述总结的独立 Markdown 导出文件。这些总结是为大模型理解做了优化，而非仅面向人类读者。

## 使用方法

### 在 SolveIt 中使用（推荐）

[SolveIt](https://solve.it.com) 是本仓库的原生使用环境。CRAFT 层级结构会自动为你提供结构化上下文。

1. 将本仓库克隆到你的 SolveIt 实例中
2. 进入任意课时文件夹（如 `part1/lesson3/`）
3. 然后：
  - 直接与课时视频内容对话
  - 或新建对话进行自主探索——CRAFT 上下文会自动加载

大模型将拥有完整的课时拆解、上下文关联与相关资源，你可以自由提问、探究原理、跨课时寻找规律，并深入任意细节。

**技巧：练习 Notebook**
可以让 SolveIt 处理官方课程练习 Notebook，将答案代码替换为分步实现提示，从而在不看答案的情况下，获得从零实现的引导式练习。

**技巧：用你的母语学习**
你可以将对话/Notebook 翻译成自己的语言，用母语与课程内容交互，理解更轻松。

### 在 SolveIt 外使用

即使不使用 SolveIt，内容依然可用：

- 上课前将 `summaries/lesson3.md` 上传给大模型做预习，或课后上传用于梳理疑问
- 在编辑器（如安装了 Copilot 的 VS Code）中打开 CRAFT 文件（标准 `.ipynb`），让大模型在你编程时拥有完整上下文

## 上下文长度注意事项

每节课完整的 CRAFT 链（根目录 + 课程部分 + 课时）大约占用：

| 阶段 | 预估 Token 数 |
|------|--------------|
| 对话开始（仅总结，不含文稿、无提示） | ~20–30k |
| 对话结束（含总结、文稿、无提示） | ~50–70k |

如果长对话中上下文过长难以处理：

- **精简周边总结**：学到第 5 课深处时，可能不再需要完整的第 5 课总结。让大模型只保留与当前位置相关的内容。
- **隐藏章节**：SolveIt 支持可折叠标题与隐藏章节，把已学内容从上下文中隐藏。
- **新建对话**：同一节课的不同模块，分开新建对话，CRAFT 上下文会干净重新加载。
- **进一步精简总结**：让 SolveIt 或其他大模型把总结压缩得更简洁。

我还在探索这类总结最合适的粒度，欢迎提出建议！

## 视频帧截取

课时 CRAFT 文件中包含 `fetch_frame()` 调用，可在指定时间戳从课程视频中截取画面。
**所有帧已提前截取并存储在 Notebook 中**，你通常不需要自己运行这些函数。

但如果你想在其他时间戳截取画面，或重新执行截取，需要在 SolveIt 与本地机器之间建立 SSH 隧道。`fetch_frame()` 通过 SSH 连接到本地机器，调用 `yt-dlp` 和 `ffmpeg` 完成截取。

### 本地环境依赖

安装所需工具：

```bash
# macOS
brew install yt-dlp ffmpeg bore-cli tmux
```

### 配置步骤

1. **在 SolveIt 上生成 SSH 密钥**，并将公钥添加到本地机器：

```bash
# SolveIt 上
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub

# 本地机器 — 粘贴公钥
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
```

2. **在本地启动 bore 隧道** 暴露 SSH 服务：

```bash
tmux new -s bore
bore local 22 --to bore.pub
# 记下返回的端口号（如 14151）
```

3. **在课时 CRAFT.ipynb 中更新端口**——每个文件顶部都有 `PORT` 变量：

```python
VIDEO_ID = '8SF_h3xF3cE'
PORT = 14151  # ← 你的 bore 端口
```

4. **从 SolveIt 测试连接**：

```bash
ssh -o StrictHostKeyChecking=no YOUR_USERNAME@bore.pub -p YOUR_PORT "echo Connection successful!"
```

更详细的步骤见 [从 SolveIt 隧道连接到本地机器](https://forbo7.github.io/forblog/posts/33_tunneling_from_solve_it_to_your_machine.html)。

#### `fetch_frame` 与 `fetch_frames` 源码

```py
import base64, re
from io import BytesIO
from PIL import Image

def fetch_frame(
    id:str,        # 视频 ID
    port:int,      # SSH 端口
    timestamp:int, # 时间戳（秒）
    src:str='yt'   # 'yt' 或 'wistia'
) -> Image.Image: # 帧图片（PIL）
    """通过 SSH 从视频中截取一帧。"""
    cmd = f'''url=$({_ytdlp_cmd(id, src)}) ffmpeg -ss {timestamp} -i "$url" -vframes 1 -f image2pipe -vcodec mjpeg - 2>/dev/null | base64'''
    result = mac(cmd, port=port)
    img_bytes = base64.b64decode(result.stdout)
    return Image.open(BytesIO(img_bytes))

def mac(
    cmd:str, # 要执行的 shell 命令
    port:int, # SSH 端口
    user:str='', # SSH 用户名
    host:str='bore.pub', # SSH 服务器
    args:str='', # 额外 SSH 参数
    timeout:int=30 # 超时秒数
) -> subprocess.CompletedProcess :
    '''在用户本地机器上执行 shell 命令。'''
    full_cmd = f"echo '{cmd}' | ssh {args} -o StrictHostKeyChecking=no -A -p {port} {user}@{host} '$SHELL -ls'"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
```

`mac` 函数基于 Jeremy 在 SolveIt 课程第 7 课中的实现。

### 本地运行（不使用 SolveIt）

如果直接在本地运行 Notebook，则无需 SSH 隧道。`yt-dlp` 和 `ffmpeg` 已在本地环境。但由于 `fetch_frame()` 是为通过 SSH 执行命令编写的，你可以将 `mac()` 调用替换为直接的 `subprocess.run()`。

## 已知问题与练习任务

仓库中仍有一些不完善的地方，非常适合作为贡献或练习任务：

### 1. Markdown 中的 KaTeX 渲染

`summaries/*.md` 中使用 `$...$` 表示数学公式，而非 `\(...\)` 和 `\[...\]`，导致公式在 SolveIt 中无法正常渲染。文稿本身的公式是正确渲染的。

**练习**：将所有 `.md` 文件中的数学公式分隔符替换为 `\(...\)`（行内公式）和 `\[...\]`（独立公式）。

### 2. 标题粒度

部分笔记单元格同时包含标题与正文。为了在 SolveIt 和 Jupyter 中更好地折叠与展开，标题应单独放在一个单元格中。

**练习**：遍历所有 CRAFT 文件，将同时包含标题和正文的单元格拆分为两个独立单元格：标题一个，内容一个。

### 3. 缺失数据伦理课程

完成本仓库后才发现，遗漏了由 [Rachel Thomas](https://rachel.fast.ai/about.html) 主讲的 [Bonus 课 8a](https://course.fast.ai/Lessons/lesson8a.html)。

**练习**：为这节课创建精读对话/Notebook 与总结，并同步更新第 1 部分总结、课程总总结、第 8 课与第 9 课总结。

### 4. 章节粒度

部分章节可能同时包含多个独立主题。更细的章节拆分能让单段内容更易于跳转与定位。

**练习**：识别包含多主题的章节，将其拆分为聚焦、可独立访问的片段。

## 相关链接

- [SolveIt](https://solve.it.com)
- [fast.ai 课程](https://course.fast.ai)
- [fastbook](https://github.com/fastai/fastbook)（免费 Jupyter Notebook 版本，也可尝试精读）
- [第 1 部分课程 Notebook](https://github.com/fastai/course22)
- [第 2 部分课程 Notebook](https://github.com/fastai/course22p2)
- [第 2 部分扩散模型 Notebook](https://github.com/fastai/diffusion-nbs)
- [数据伦理 Bonus 课 8a](https://course.fast.ai/Lessons/lesson8a.html)
- [fast.ai 论坛](https://forums.fast.ai)

## 致谢

本项目基于 Jeremy Howard 与 Rachel Thomas 的 [《程序员实用深度学习》](https://course.fast.ai) 构建。仓库中所有对话与 Notebook 均源自他们的课程视频与官方页面。本仓库将其内容重新组织，用于结构化精读与上下文管理，帮助学习者结合大模型进行更深入的学习。

---

*由 [Salman Naqvi](https://forbo7.github.io) 使用 [SolveIt](https://solve.it.com) 制作。*
