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
*   **LLM 处理:**
    *   基于选定的模板动态生成提示词 (Prompt)。
    *   调用本地 Qwen-VL 模型进行 OCR 和信息提取。
*   **结果预览与导出:**
    *   在界面中预览 LLM 的原始输出。
    *   尝试根据模板提示将结果解析为表格数据。
    *   支持将提取结果导出为 CSV, XLSX, 或 JSON 格式 (合并导出或按文件单独导出)。
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
│   ├── preprocessor.py     # 模板选择/编辑
│   ├── processor.py        # LLM 处理
│   ├── formatter.py        # 结果格式化/导出
│   └── model_loader.py     # 模型加载
├── templates/              # 预定义模板 (JSON)
├── utils/                  # 通用工具函数
├── assets/                 # 静态资源 (可选)
├── outputs/                # 默认导出目录 (.gitignore 包含)
├── requirements.txt        # Python 依赖
├── .streamlit/             # Streamlit 配置
│   └── config.toml
└── README.md               # 本文档
```

## 🚀 安装与运行

1.  **克隆仓库 (如果需要):**
    ```bash
    git clone <your-repository-url>
    cd LocalLlmOcr
    ```

2.  **创建虚拟环境 (推荐):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **安装依赖:**
    确保你的环境满足 PyTorch 和 CUDA 的要求 (如果使用 GPU)。然后安装 `requirements.txt` 中的包：
    ```bash
    pip install -r requirements.txt
    ```
    *注意:* `flash-attn` 的安装可能需要特定的编译环境或 CUDA 版本。

4.  **运行应用:**
    ```bash
    streamlit run streamlit_app.py
    ```
    应用将在你的浏览器中打开 (通常是 `http://localhost:8501`)。
    
    *注意:* 如果遇到与 PyTorch 相关的错误，可以尝试禁用文件监视器：
    ```bash
    echo "[server]\nfileWatcherType = \"none\"" > .streamlit/config.toml
    ```

## 📖 使用说明

1.  **登录:** 打开应用后，会显示一个简单的登录界面。输入任意用户名和密码即可进入主应用 (此功能为占位符)。
2.  **上传文件:** 在左侧边栏，点击"浏览文件"按钮，选择一个或多个图片文件上传。
3.  **选择/编辑模板:**
    *   从下拉菜单中选择一个预设模板 (如 `invoice_template`)。
    *   或选择"手动创建/编辑新模板"来定义你自己的提取规则。
    *   (可选) 展开"微调模板"部分，使用表格编辑器添加、删除或修改字段名称、格式提示和是否必须。
    *   (可选) 如果你修改了模板并想保存，在"另存为新模板名称"中输入名字，然后点击"保存模板"。
4.  **开始处理:** 确认已上传文件并选择了模板后，点击侧边栏底部的"🚀 开始处理"按钮。
5.  **查看结果:** 应用主区域将显示每个文件的处理状态和 LLM 的原始输出。
6.  **下载结果:**
    *   在结果预览下方，你可以找到下载选项。
    *   **合并下载:** 将所有成功处理的表格数据合并下载为 CSV, XLSX, 或 JSON 文件。
    *   **单独下载:** 展开每个文件的下载选项，可以单独下载该文件的 CSV, XLSX, JSON 或原始文本。
7.  **退出登录:** 点击侧边栏顶部的"退出登录"按钮返回登录界面。

## ⚙️ 配置

*   **`.streamlit/config.toml`**: 此文件用于配置 Streamlit。目前包含 `[server] fileWatcherType = "none"` 来禁用文件监视器，以解决与 PyTorch 可能存在的冲突。禁用后，修改代码需要手动重启 Streamlit 服务才能生效。

## 🧪 从单文件应用升级

本项目是从 `streamlit_ocr_app.py` 单文件应用升级而来的模块化版本：

*   **原应用:** 提供单文件处理、固定提示词的简单发票OCR功能。
*   **新应用:** 支持多文件处理、模板选择与自定义、多种输出格式，以及完整的UI流程。

## 🔮 未来工作

*   实现真正的用户认证和数据库集成，以存储用户特定的模板。
*   集成 LLM 智能推荐模板字段的功能。
*   优化 LLM 提示词和输出解析逻辑，提高准确性和鲁棒性。
*   增加对 PDF 等更多文件类型的支持。
*   添加批量处理队列，提高处理大量文件时的效率。
*   优化界面，增加更多的数据可视化选项。