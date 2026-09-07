# 最终改进版的裸机验收

当前状态见 [实施状态](implementation-status.md)。本文是先前准备的执行记录模板，存在这些步骤不代表资源验收通过。用户最新要求本次在 baseline 完成后结束，部署已暂停，需先重新明确 [后续边界](session-boundary.md)；B0 仅本地评测。

## 部署范围

目标主机已核实为 Ubuntu 24.04.4、2 核、物理内存 2,063,233,024 bytes、swap 2,084,564,992 bytes。已有其他服务，应把它们与操作系统的占用一起计入整机指标。18443 对应旧 nginx 站点和旧 Trust RAG 后端已关闭，其余服务保持原状态。

新服务使用独立目录和 systemd 单元。部署前确认最终冻结的 stage/without、代码提交、语料及向量清单。API key 只进入权限为 600 的运行环境文件，不进入 Git、命令输出或评测报告。Qdrant 和问答 API 默认只监听回环地址。

原始 Office 文档在本地离线转换为归一化 Evidence；Linux 从该 Evidence 执行完整导入。转换清单保留原始文件及输出哈希。在线应用和 Linux 导入不依赖 Microsoft Word。

## 维护期导入

1. 确认问答服务已停止，记录主机、CPU、磁盘、内核及 Hatch/Python/Qdrant 版本。
2. 在新的运行数据目录执行 `prepare`，保留 `corpus-resources.json` 和 `corpus.manifest.json`。原始 Evidence 和模型配置必须与本地最终实验一致。
3. 使用已完成的 float32 embedding 缓存验证全语料覆盖，记录 `embedding-build.json`。若需要重新生成向量，调用原先授权的外部 API 并单独记录费用。
4. 向空的本地 Qdrant collection 执行 `index-qdrant`，记录写入、索引优化和整机 RAM/swap 峰值。等待优化完成后才能启动问答。
5. 检查 OOM 计数增量、完整点数及服务日志。连接超时、重试和中断均保留，不能只报告最终一次成功。

整机监控应覆盖维护流程开始到索引优化完成；各 CLI 的自身 RSS 不包含独立 Qdrant 进程。缓存复制与首次外部向量生成应分开报告，不能把缓存复用时间解释为首次建库时间。

## 冷启动与双并发

停止应用和 Qdrant 后重新启动，记录到健康检查成功的时间、整机内存和 swap。仅重启进程不等于清空操作系统文件缓存，应明确冷启动的实际条件。

应用启动后，用开发题组成覆盖五类的持续双并发负载。例如在项目目录运行：

```sh
hatch -e trust-rag run python -m applications.trust_rag.tools.acceptance \
  --cases RUNTIME_DATA/splits/dev.jsonl --per-type 4 --rounds 3 \
  --url http://127.0.0.1:8000 --output applications/trust_rag/outputs/linux-warm-01
```

该工具固定两个请求并发，只把问题发送到服务，答案标签留在离线评分端；输出逐次状态、P50/P95、整机 RAM/swap、换页量和 OOM 计数。重复请求用于资源稳定性检查，不是新的独立正确率样本。它不会自行宣称完整验收通过。

最终判定应联合维护期导入、优化、冷启动和持续查询四类记录，明确实际持续时间、错误及资源峰值。目标是这台约 2 GiB RAM + 2 GiB swap 的机器能够完成流程并持续运行。Windows 大内存机器的结果不能替代此判定。
