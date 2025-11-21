1. 技术栈
   1. 使用Python作为编程语言
   2. 使用uv作为项目管理工具, 并使用现代的项目管理方式, 所有的 Python 操作都通过 uv 管理（包括运行脚本、安装包、创建环境等），请在提供建议时只使用 uv 对应的命令，而不是传统的 python/pip/venv 等命令。
      *   **项目初始化**: 使用 `uv init` 创建项目的核心配置文件 `pyproject.toml`。
      *   **创建虚拟环境**: 使用 `uv venv` 快速创建一个独立的 `.venv` 环境来隔离依赖。
      *   **添加生产依赖**: 使用 `uv add <package>` 将项目运行所需的包（如 `pandas`, `fastapi`）安装到环境并记录到 `pyproject.toml`。
      *   **添加开发依赖**: 使用 `uv add <package> --dev` 添加仅在开发、测试时使用的工具（如 `pytest`, `ruff`）。
      *   **执行命令**: 使用 `uv run <command>` (例如 `uv run python main.py`) 来执行脚本，无需手动激活虚拟环境。
      *   **生成锁文件**: 使用 `uv lock` 来解析所有依赖的精确版本，并生成 `uv.lock` 文件以确保环境的可复现性（此文件必须提交到Git）。
      *   **同步环境**: 使用 `uv sync` 命令，让团队成员或服务器可以根据 `uv.lock` 文件完美地、精确地复制你的开发环境。
      *   **代码检查与格式化**: 使用 **Ruff** 工具，并将它的配置统一写在 `pyproject.toml` 的 `[tool.ruff]` 部分。
      *   **运行代码检查与格式化**: 使用 `uv run ruff check .` 来检查代码问题，使用 `uv run ruff format .` 来自动格式化代码。
      *   **打包配置**: 在 `pyproject.toml` 文件中，详细填写 `[project]` 部分（项目名称、版本、作者等）和 `[build-system]` 部分。
   3. 使用powershell作为终端, 不要使用Linux以及类Unix系统的命令
      1. 单条命令聚焦单一任务, 不要使用复合命令
      2. 由于PowerShell 5.1不支持&&操作符，使用分号分隔命令来初始化uv项目并安装依赖。
7. 工具使用
   1. 使用todolist工具来创建一个工作列表





