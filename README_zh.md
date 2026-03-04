# fastai-close-reading

![fastai-close-reading](thumbnail.png)

对 Jeremy Howard 主讲的 [Practical Deep Learning for Coders](https://course.fast.ai) 全部课程进行**结构化精读**（更准确地说是“精看”）——包含第 1 部分和第 2 部分共 28 节课的逐课字幕与笔记。

不包含 [数据伦理 bonus 课 8a](https://course.fast.ai/Lessons/lesson8a.html)。欢迎你自行补充！

本仓库主要为在 [SolveIt](https://solve.it.com) 上进行**上下文学习**而设计，但也可配合其他工具使用。

[English README →](README.md)

---

## 这是什么？

本仓库包含 fast.ai《Practical Deep Learning for Coders》课程每一节课的字幕与结构化笔记，格式为 Jupyter Notebook。
每节课都被拆分为多个片段，包含：

- **讲解内容**：Jeremy 与团队所说、所演示的完整叙事
- **视频帧引用**：通过 `fetch_frame()` 调用捕获幻灯片、图表、实操演示画面
- **上下文关联**：自动加载上一课、当前课、下一课的总结（在 SolveIt 中会自动加载）
- 按课、按部分整理的**挑战、作业、研究方向与资源**

目标是**从课程内容中“读出来”**，而不是“钻进去”。
帮助你成为更具批判性的学习者，更深度地理解内容，并保持心流状态。
关于精读的更多说明见[这里](https://www.fast.ai/posts/2026-01-21-reading-LLMs/)。

## 仓库结构

```
.
├── CRAFT.ipynb                     # 课程总览总结
├── CONTROLLER.ipynb                # 管理面板（课程目录）
├── part1/
│   ├── CRAFT.ipynb                 # 第 1 部分总结、挑战、资源
│   ├── lesson0/ → lesson8/
│   │   └── CRAFT.ipynb             # 单课：总结 + 精读笔记
├── part2/
│   ├── CRAFT.ipynb                 # 第 2 部分总结、挑战、资源
│   ├── lesson9/ → lesson25/
│   │   └── CRAFT.ipynb             # 单课：总结 + 精读笔记
└── summaries/
    ├── all.md                      # 全课程合并总结
    ├── part1.md / part2.md         # 分部分总结
    └── lesson0.md ... lesson25.md  # 单课独立总结
```

### 什么是 CRAFT 文件？

在 [SolveIt](https://solve.it.com) 中，**CRAFT.ipynb** 为每个文件夹提供专属的 LLM 上下文。
当你在某个课程文件夹内打开对话时，SolveIt 会自动加载该文件夹及所有父文件夹的 CRAFT 文件，形成**层级上下文**：

| 层级 | 内容 |
|------|------|
| **根目录** | 完整课程总览 |
| **分部分**（`part1/`、`part2/`） | 部分总结 + 挑战、作业、研究方向、资源 |
| **单课**（`part1/lesson3/` 等） | 上一课 + 当前课 + 下一课总结、视频 ID、带视频帧的逐段精读笔记 |

打开 `part1/lesson3/` 后，你会在第一次提问前就自动获得：
课程总览 → 第 1 部分总结 → 第 3 课完整精读笔记。

### `summaries/` 文件夹

每节课叙事总结的独立 Markdown 导出文件。
我对这些总结做了专门优化，**对 LLM 比对人类读者更友好**。

## 使用方法

### 在 SolveIt 中使用（推荐）

[SolveIt](https://solve.it.com) 是本仓库的原生运行环境。
CRAFT 层级结构会自动为你提供结构化上下文。

1. 将本仓库克隆到你的 SolveIt 实例中
2. 进入某节课文件夹（如 `part1/lesson3/`）
3a. 直接与本节课视频内容对话
3b. 或新建对话进行自主探索——CRAFT 上下文会自动加载

LLM 将拥有完整的课程拆解、上下文与资源，你可以：
提问、探究含义、跨课寻找规律、对任意片段进行深度挖掘。

**技巧：练习笔记本**
让 SolveIt 处理官方课程练习笔记本，把答案代码替换为步骤说明注释，
实现“从零手写、无答案提示”的引导式练习。

**技巧：用你的母语学习**
可以将对话/笔记本翻译成你的语言，用母语与课程内容交互，理解更顺畅。

### 不使用 SolveIt

即使没有 SolveIt，内容依然可用：

- 上课前上传 `summaries/lesson3.md` 预习，或课后复习梳理疑点
- 在编辑器（如安装了 Copilot 的 VS Code）中打开 CRAFT 文件（标准 `.ipynb`）
  让 LLM 在你写代码时拥有完整上下文

## 上下文长度说明

每节课的完整 CRAFT 链（根目录 + 分部分 + 单课）大致占用：

| 阶段 | Token 估算 |
|------|------------|
| 对话开始（仅总结，不含字幕，无提问） | ~20–30k |
| 对话结束（含总结 + 字幕，无提问） | ~50–70k |

如果长对话中上下文过长难以处理：

- **精简周边总结**：学到第 5 课深处时，可能不再需要完整的第 5 课总结。
  让 LLM 只保留与当前位置相关的内容。
- **折叠/隐藏章节**：SolveIt 支持可折叠标题与隐藏章节。
  把已学内容从上下文中隐藏。
- **新开对话**：同一课的不同片段，分开新建对话。
  每次都会干净重新加载 CRAFT 上下文。

## 视频帧捕获

课程 CRAFT 文件中包含 `fetch_frame()` 调用，用于在指定时间戳捕获课程视频截图。
**所有帧已经提前捕获并保存在笔记本中**——你不需要自己运行这些调用。

但如果你想在其他时间戳捕获画面，或重新捕获，
需要从 SolveIt 到本地机器建立 SSH 隧道。
`fetch_frame()` 和 `fetch_frames()` 会通过 SSH 连接到你的机器，执行 `yt-dlp` 和 `ffmpeg`。

### 本地环境依赖

安装所需工具：

```bash
# macOS
brew install yt-dlp ffmpeg bore-cli tmux
```

### 设置步骤

1. **在 SolveIt 上生成 SSH 密钥**，并将公钥添加到本地机器：

```bash
# SolveIt 上
ssh-keygen -t ed25519
cat ~/.ssh/id_ed25519.pub

# 本地机器 — 粘贴公钥
echo "YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
```

2. **在本地启动 bore 隧道** 暴露 SSH：

```bash
tmux new -s bore
bore local 22 --to bore.pub
# 记下返回的端口号（如 14151）
```

3. **在对应课程的 `CRAFT.ipynb` 中更新端口**
   每节课顶部都有 `PORT` 变量：

```python
VIDEO_ID = '8SF_h3xF3cE'
PORT = 14151  # ← 你的 bore 端口
```

4. **从 SolveIt 测试连接**：

```bash
ssh -o StrictHostKeyChecking=no YOUR_USERNAME@bore.pub -p YOUR_PORT "echo Connection successful!"
```

更详细的步骤见：[Tunneling from SolveIt to your Machine](https://forbo7.github.io/forblog/posts/33_tunneling_from_solveit_to your_machine.html)。

### 本地运行（不使用 SolveIt）

如果直接在本地运行笔记本，则不需要 SSH 隧道——`yt-dlp` 和 `ffmpeg` 已经在本地。
但 `fetch_frame()` 函数是为通过 SSH 执行而写的。你有两种选择：

1. **不运行这些函数**——所有帧已经捕获并保存在笔记本里。
2. **替换为本地执行**：把 `mac()` 调用换成直接的 `subprocess.run()`：

```python
import subprocess, base64
from PIL import Image
from io import BytesIO

def fetch_frame_local(id, timestamp, src='yt'):
    url_cmd = f'yt-dlp -f "best[height<=720]" -g "https://youtu.be/{id}" | head -1'
    url = subprocess.run(url_cmd, shell=True, capture_output=True, text=True).stdout.strip()
    cmd = f'ffmpeg -ss {timestamp} -i "{url}" -vframes 1 -f image2pipe -vcodec mjpeg - 2>/dev/null'
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return Image.open(BytesIO(result.stdout))
```

## 已知问题与练习

本仓库仍有一些不完善的地方。
如果你想贡献或练习，这些都是很好的练习项目。

### 1. Markdown 文件中的 KaTeX 渲染

`summaries/*.md` 使用 `$...$` 表示数学公式，而非 `\(...\)` 和 `\[...\]`。
因此公式在 SolveIt 中无法正常渲染。字幕本身的公式已正确渲染。

**练习**：将所有 `.md` 文件中的数学公式分隔符替换为：
- 行内公式：`\(...\)`
- 块级公式：`\[...\]`

### 2. 标题粒度

部分笔记单元格同时包含标题和正文。
为了在 SolveIt 和 Jupyter 中更好地折叠，标题应单独放在一个单元格。

**练习**：遍历所有 CRAFT 文件，
将“标题+正文”的单元格拆分为两个独立单元格：标题一个，内容一个。

### 3. 缺失数据伦理课程

完成本仓库后我才发现，遗漏了由 [Rachel Thomas](https://rachel.fast.ai/about.html) 主讲的 [Bonus Lesson 8a](https://course.fast.ai/Lessons/lesson8a.html)。

**练习**：为这节课创建精读对话/笔记本与总结，
并同步更新第 1 部分总结、全课程总结、第 8 课和第 9 课的总结。

### 4. 章节粒度

部分章节同时覆盖多个独立主题。
更细的分段会让跳转与定位更轻松。

**练习**：识别多主题章节，
将其拆分为聚焦、可独立访问的小片段。

## 相关链接

- [SolveIt](https://solve.it.com)
- [fast.ai 课程](https://course.fast.ai)
- [fastbook](https://github.com/fastai/fastbook)（免费 Jupyter 笔记本；也可以尝试精读它）
- [第 1 部分课程笔记本](https://github.com/fastai/course22)
- [第 2 部分课程笔记本](https://github.com/fastai/course22p2)
- [第 2 部分扩散模型笔记本](https://github.com/fastai/diffusion-nbs)
- [数据伦理 Bonus 课 8a](https://course.fast.ai/Lessons/lesson8a.html)
- [fast.ai 论坛](https://forums.fast.ai)

## 致谢

基于 Jeremy Howard 和 Rachel Thomas 的 [Practical Deep Learning for Coders](https://course.fast.ai) 构建。
本仓库中所有对话与笔记本均源自他们的课程视频与官网页面。
本仓库对内容进行重构，用于结构化精读与上下文组织，以便结合 LLM 进行深度学习。

---

*由 [Salman Naqvi](https://forbo7.github.io) 使用 [SolveIt](https://solve.it.com) 制作。*
