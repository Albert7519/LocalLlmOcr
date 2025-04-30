# 本地 LLM OCR 应用

这是一个使用本地部署的大型语言模型 (LLM) Qwen-VL 进行光学字符识别 (OCR) 和信息提取的 Streamlit 应用。
该应用经过重构，具有模块化、可扩展的特点，支持处理多种文档类型和输出格式。

## ✨ 功能特性

*   **多文件上传:** 支持同时上传和处理多个图片文件 (JPG, JPEG, PNG)。
*   **模板化提取:**
    *   提供预设模板 (如发票、财务报表)。
    *   允许用户选择、编辑现有模板或创建新模板。
    *   通过交互式界面 (`st.data_editor`) 动态调整需要提取的字段、格式要求等。
    *   保存和加载自定义模板 (存储为 JSON 文件)。
*   **AI 智能推荐模板:** 
    *   使用 LLM 分析图片内容，自动识别关键字段。
    *   智能生成字段结构和文档类型推荐。
    *   临时模板自动保存，可直接用于信息提取。
    *   支持基于 AI 推荐的模板编辑与永久保存。
*   **LLM 处理:**
    *   基于选定的模板动态生成提示词 (Prompt)。
    *   调用本地 Qwen-VL 模型进行 OCR 和信息提取。
*   **结果预览与导出:**
    *   在界面中预览 LLM 的原始输出。
    *   尝试根据模板提示将结果解析为表格数据。
    *   支持将提取结果导出为 CSV, XLSX, 或 JSON 格式 (合并导出或按文件单独导出)。
    *   智能表格解析，能够正确处理 Markdown 表格和 CSV 格式数据。
*   **模块化设计:** 代码被拆分为独立的模块 (文件处理、预处理、核心处理、格式化输出、模型加载、认证)。
*   **用户认证 (占位符):** 包含一个简单的登录界面框架，为未来实现用户特定功能预留空间。

## 🛠️ 技术栈

*   **框架:** Streamlit
*   **语言:** Python 3.x
*   **LLM:** Qwen-VL (通过 `modelscope` / `transformers`)
*   **核心库:** PyTorch, Pandas, Pillow, openpyxl
*   **依赖管理:** `requirements.txt`

## 📂 项目结构

```
LocalLlmOcr/
├── streamlit_app.py        # 主应用入口
├── modules/                # 各功能模块
│   ├── auth.py             # 认证 (占位)
│   ├── file_handler.py     # 文件处理
│   ├── preprocessor.py     # 模板选择/编辑与AI推荐
│   ├── processor.py        # LLM 处理
│   ├── formatter.py        # 结果格式化/导出
│   └── model_loader.py     # 模型加载
├── templates/              # 预定义模板 (JSON)
├── utils/                  # 通用工具函数
│   └── helpers.py          # 模板加载/保存/管理
├── assets/                 # 静态资源 (可选)
├── outputs/                # 默认导出目录 (.gitignore 包含)
├── requirements.txt        # Python 依赖
├── .streamlit/             # Streamlit 配置
│   └── config.toml
└── README.md               # 本文档
```

## 🚀 安装与运行

### 通用步骤

1.  **克隆仓库 (如果需要):**
    ```bash
    git clone <your-repository-url>
    cd LocalLlmOcr
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\\Scripts\\activate  # Windows
    ```

### 针对 NVIDIA GPU (Linux/Windows)

1.  **安装依赖:**
    确保你的环境满足 PyTorch 和 CUDA 的要求。然后安装 `requirements.txt` 中的包：
    ```bash
    pip install -r requirements.txt
    ```
    *注意:* `flash-attn` 的安装可能需要特定的编译环境或 CUDA 版本。如果遇到问题，可以尝试移除它，模型会回退到标准注意力实现。

2.  **运行应用:**
    ```bash
    streamlit run streamlit_app.py
    ```

### 针对 macOS (Apple Silicon M 系列芯片)

1.  **安装依赖:**
    *   **安装 PyTorch for Mac:** 访问 [PyTorch 官网](https://pytorch.org/) 获取适合你 Mac 的安装命令。通常是：
        ```bash
        pip install torch torchvision torchaudio
        ```
    *   **安装其他依赖:** `flash-attn` 已从 `requirements.txt` 中移除，因为它不兼容 macOS。运行：
        ```bash
        pip install -r requirements.txt
        ```

2.  **运行应用:**
    ```bash
    streamlit run streamlit_app.py
    ```
    *   **说明:** 此版本已修改为自动检测并使用 Mac 的 Metal Performance Shaders (MPS) 后端进行 GPU 加速。无需 `flash-attn`。性能通常会低于在相当的 NVIDIA GPU 上使用 CUDA 的情况，但优于纯 CPU 运行。

### 应用启动后

应用将在你的浏览器中打开 (通常是 `http://localhost:8501`)。

*注意:* 如果遇到与 PyTorch 相关的错误 (在任何系统上)，可以尝试禁用文件监视器：
```bash
# 在项目根目录下创建或修改 .streamlit/config.toml 文件
echo "[server]\\nfileWatcherType = \\"none\\"" > .streamlit/config.toml
# 然后重新运行 streamlit
```
禁用后，修改代码需要手动重启 Streamlit 服务才能生效。

## 📖 使用说明

1.  **登录:** 打开应用后，会显示一个简单的登录界面。输入任意用户名和密码即可进入主应用 (此功能为占位符)。
2.  **上传文件:** 在左侧边栏，点击"浏览文件"按钮，选择一个或多个图片文件上传。
3.  **AI 智能推荐模板:**
    *   上传图片后，点击"🔍 AI 分析图片并推荐模板"按钮。
    *   系统将使用大模型分析图片内容，自动识别文档类型和关键字段。
    *   分析完成后，会自动创建临时模板并选择它，可继续进行处理或编辑。
4.  **选择/编辑模板:**
    *   从下拉菜单中选择一个预设模板 (如 `invoice_template`)，或使用 AI 推荐的模板。
    *   或选择"手动创建/编辑新模板"来定义你自己的提取规则。
    *   (可选) 展开"微调模板"部分，使用表格编辑器添加、删除或修改字段名称、格式提示和是否必须。
    *   (可选) 如果你修改了模板并想保存，在"另存为新模板名称"中输入名字，然后点击"保存模板"。
5.  **开始处理:** 确认已上传文件并选择了模板后，点击侧边栏底部的"🚀 开始处理"按钮。
6.  **查看结果:** 应用主区域将显示每个文件的处理状态和 LLM 的原始输出。
7.  **下载结果:**
    *   在结果预览下方，你可以找到下载选项。
    *   **合并下载:** 将所有成功处理的表格数据合并下载为 CSV, XLSX, 或 JSON 文件。
    *   **单独下载:** 展开每个文件的下载选项，可以单独下载该文件的 CSV, XLSX, JSON 或原始文本。
8.  **退出登录:** 点击侧边栏顶部的"退出登录"按钮返回登录界面。

## ⚙️ 配置

*   **`.streamlit/config.toml`**: 此文件用于配置 Streamlit。目前包含 `[server] fileWatcherType = "none"` 来禁用文件监视器，以解决与 PyTorch 可能存在的冲突。禁用后，修改代码需要手动重启 Streamlit 服务才能生效。

## 🧪 应用发展历程

本项目经历了以下发展阶段：

*   **单文件应用 (`streamlit_ocr_app.py`):** 初始版本，提供单文件处理、固定提示词的简单发票OCR功能。
*   **模块化重构:** 将应用拆分为多个功能模块，支持多文件处理、模板选择与自定义、多种输出格式。
*   **AI 智能推荐功能:** 添加了使用大模型自动分析图片内容并推荐字段结构的能力，提高了应用的智能化水平。
*   **表格解析优化:** 加强了对Markdown表格和各种格式数据的解析能力，提升了XLSX导出的质量。

## 🔮 未来工作

*   实现真正的用户认证和数据库集成，以存储用户特定的模板。
*   进一步优化 AI 智能推荐功能，支持更多文档类型的智能识别。
*   优化 LLM 提示词和输出解析逻辑，提高准确性和鲁棒性。
*   增加对 PDF 等更多文件类型的支持。
*   添加批量处理队列，提高处理大量文件时的效率。
*   优化界面，增加更多的数据可视化选项。
*   添加自定义 LLM 模型选择功能，支持更多的视觉语言模型。