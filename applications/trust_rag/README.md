# Trust RAG on Haystack

本应用位于 Haystack 源码仓库的 `applications/trust_rag`，不依赖旧 Trust RAG 的 Python 包。上游固定为 Haystack v3.1.1，提交 `d60cce01a778bc3498a02866fe288b3c07398649`。优先采用组件与 Pipeline 扩展，未修改 Haystack 核心代码。

## 环境

先确保 `hatch --version` 可用，然后在仓库根目录运行：

```sh
hatch -e trust-rag run lab --help
hatch -e trust-rag run test
hatch -e trust-rag run lint
```

`trust-rag` 是仅包含本应用依赖的 Hatch 环境，避免安装本地 torch/transformers 模型栈。具体依赖以根目录 `pyproject.toml` 为准。Qdrant Server 固定 1.19.1，Python client 固定 1.19.0。

## 数据准备和模型调用

复制 `settings.example.json` 到个人配置文件，设置 `data_dir`。所有数据和索引默认位于仓库根 `.lab-data/`，被 Git 忽略。设置环境变量 `OPENAI_API_KEY`，或通过 `--env-file` 指定本地密钥文件；不要将密钥提交到仓库。

```sh
hatch -e trust-rag run lab prepare --evidence /path/to/evidence.jsonl
hatch -e trust-rag run lab split --cases /path/to/mcq_cases.jsonl
hatch -e trust-rag run lab --env-file /path/to/local.env smoke-api
hatch -e trust-rag run lab --env-file /path/to/local.env embed --limit 320
hatch -e trust-rag run lab estimate
hatch -e trust-rag run lab --env-file /path/to/local.env embed --workers 4
```

`prepare` 流式读取旧 Evidence JSONL，通过明确字段白名单生成统一文本、证据位置和 SQLite FTS5 BM25。表格从 baseline 起即参与检索；评测答案、题型标签和预期引用不会被写入在线索引。转换结果与输入哈希一并落盘。

`split` 将关联来源、近重复问题分组，确保五种 qa_type 在两集合中都有覆盖。已生成划分为开发集 208 题、测试集 90 题，按来源与难度划分的九个交叉层也都有覆盖。清单包含固定 seed、题目 ID 分组和文件哈希；重复运行会核验既有划分而不是静默改写。

`embed` 只调用外部 API，向量缓存键包含服务、模型和输入文本。中断后可重跑，已有缓存不重复请求；全量缓存未完成时正式索引加载会报错。`--limit` 仅适用于局部兼容性检查，不是全语料成绩。

`api-usage.jsonl` 保存调用时间、模型、token 和重试记录，不保存密钥。启动持久化日志前的中断任务不能补造精确费用；实际费用应以服务账单为准，不能把缺失金额视作零费用。

## Qdrant 与实验

旧 `full_no_legacy_doc` 索引不包含 `.doc`。完整实验先用本机 Word 的只读导出工具 `tools/export_legacy_word.ps1` 导出原始 `.doc`，再通过 `hatch -e trust-rag run python -m applications.trust_rag.tools.merge_sources --evidence OLD_EVIDENCE --legacy-word EXPORT_DIR --output NEW_EVIDENCE` 合并；合并工具不会读取 QA 工作簿或答案。随后对所有阶段统一执行 `prepare`。在线服务无需安装 Word，Linux 导入从归一化 Evidence 开始；原始 Office 解析属于本地离线预处理。保留原始文件及导出哈希可追溯转换。

语料元数据同时保留附件标题和母文件标题，来源命中指标接受两种真实来源名称。新快照应使用新的 `data_dir`，旧实验保留；向量缓存可通过关闭所有读写进程后的 SQLite 文件副本复用。

从 Qdrant 官方 v1.19.1 release 安装本机架构对应二进制，核实校验值；在仓库根启动：

```sh
qdrant --config-path applications/trust_rag/deploy/qdrant.yaml
hatch -e trust-rag run lab index-qdrant
hatch -e trust-rag run lab evaluate --stage B0 --split dev --output applications/trust_rag/outputs/dev-b0
hatch -e trust-rag run lab evaluate --stage B1 --split dev --output applications/trust_rag/outputs/dev-b1
hatch -e trust-rag run lab compare applications/trust_rag/outputs/dev-b0 applications/trust_rag/outputs/dev-b1
```

所有模型命令可加全局 `--config /path/to/settings.json --env-file /path/to/local.env`，放在子命令之前。未指定 env-file 时只使用现有进程环境，不自动读取旧仓库。

| 阶段 | 查询行为 |
|---|---|
| B0 | 原生 InMemoryDocumentStore + InMemoryEmbeddingRetriever |
| B1 | Qdrant cosine 向量检索 |
| B2 | Qdrant + jieba/SQLite FTS5 BM25，经 RRF 融合 |
| B3 | 加入模型规划的精确表格 SQL 查询及 Decimal 运算 |
| B4 | 加入引用 ID/原文检查和 API 证据支持检查 |

检索候选数、最终上下文预算和基础生成提示词保持一致。B3 只能查询本轮已经发现的 doc_id；二元运算要求两个明确且不同的单元格，集合运算不允许截断后求和。B4 的 API 检查属于被测能力，不作为评测真值。

Qdrant 导入按固定 ID 重放瞬时连接失败，并等待优化状态连续正常后才完成。比较查询性能前应确认后台优化已经结束；[Qdrant 优化器说明](https://qdrant.tech/documentation/operations/optimizer/)解释了建索引与查询资源竞争的影响。

消融示例：`evaluate --stage B4 --without tables --split dev --output ...`。`--without` 可重复，用于关闭 `bm25`、`tables` 或 `verifier`。最终组件选择先在开发集完成。

```sh
hatch -e trust-rag run lab freeze --stage B4
hatch -e trust-rag run lab evaluate --stage B0 --split test --output applications/trust_rag/outputs/test-b0
hatch -e trust-rag run lab evaluate --stage B4 --split test --output applications/trust_rag/outputs/test-b4
```

`freeze` 的 stage/without 必须对应实际选择的最终候选。保留测试集评测要求代码、模型配置、语料和划分与冻结清单一致；冻结后变更需使用新实验并注明旧测试已曝光。baseline 和所有中间阶段应在冻结协议下统一评测，不能只挑最好的测试结果。

输出保留逐题结果、错误/拒答、引用、各类准确率、检索/完整问答延迟、API token 和资源记录。拒答、错误都留在正确率分母中。缺少细粒度引用 gold 的题目仅计算来源命中，有位置标注时才计算位置命中。

## 裸机服务

```sh
hatch -e trust-rag run lab --config /etc/trust-rag.json serve --stage B4
```

提供 `GET /health`、`POST /ask`，请求体 `{"question":"问题和选项"}`。目前面向已约定的选择题评测契约。默认只监听回环地址，最多两条同时进行的问答，超出返回 429。配置 B4 前必须完成该阶段的实际评测与最终组件选择。

`deploy/` 中提供 systemd 模板。根据机器实际 Hatch 路径修改 ExecStart，以服务用户提前创建 Hatch 环境；Qdrant 和应用由同一有权限访问 data_dir 的专用用户运行。模板不会自动修改系统用户、swap 或防火墙。

目标机器为 Linux 整机 2 GiB RAM + 2 GiB swap。导入时停止查询服务，完成后重新启动。实际资源验收需要记录完整导入、索引优化、冷启动、持续双并发查询、整机 RAM/swap、OOM 与错误，以及 CPU/磁盘/版本。Windows 的进程 RSS、Qdrant local 模式和大内存机器结果不能代替真实整机验收。

## 当前实施边界

远程 fork、完整分阶段结果和目标 Linux 资源验收的实际状态以 `docs/implementation-status.md` 为准。存在代码或通过离线测试不代表最终质量与资源目标已经达成。
