# 实施状态

更新于 2026-09-07。所有本次代码、索引和评测产物均在 `D:/Code/trust-rag-v2`；旧 `D:/Code/trust-rag` 仅作语料、评测与本地 API 配置的读取来源。

## 已完成

- 检出 Haystack v3.1.1 / d60cce01a778bc3498a02866fe288b3c07398649，创建本地 trust-rag/reconstruction 分支。
- 新增 applications/trust_rag 应用、独立 Hatch 环境与 B0–B4 可开关 Pipeline。
- 转换 188,173 条 Evidence、467 份文档，得到 188,258 个固定检索片段；SQLite 文件约 420 MiB。
- 固定开发集 208 题、测试集 90 题，五类及九个来源/难度交叉层两侧均有覆盖；43 个来源/近重复联合组没有跨集合拆分。
- 外部 embedding/生成小样本 API 兼容性验证通过。embedding 返回 1024 维。全量 188,258 个向量及 Qdrant Server 索引已完成。
- 本地契约和原生 Haystack/Qdrant local 流水线、并发与评测防泄漏测试 20 项通过；客户端断开后，未完成的问答仍占用并发名额。
- 已创建 https://github.com/chaos-lsc/haystack fork，当前 v2 checkout 的 origin 指向该 fork，upstream 指向 deepset-ai/haystack。

## 正在进行

- 本地开发集阶段实验、消融与最终选择。用户要求本地测试结束后再进行远端部署；baseline 仅在本地评测，远端只部署最终改进版。

## 前置条件与未完成项

- 开发分支尚待提交和推送，完整阶段评测正在进行。
- 用户指定 Linux 主机的专用部署密钥已配置并验证。系统为 Ubuntu 24.04.4、2 核、物理内存 2,063,233,024 bytes、swap 2,084,564,992 bytes。已有其他服务，真实验收需包括其资源占用。连接标识仅保存在本地忽略文件中。
- 按用户要求，18443 对应 nginx server block 已关闭，后端 trust-rag.service 已停止，18443/8000 无监听。nginx 原配置备份为 /etc/nginx/trust-rag-review.before-stop-20260907。其他站点不受影响。
- 远端准备已暂停；已传输的源码/数据包留在 ~/trust-rag-v2，尚未启动新应用。继续部署需等待本地测试结束。
- 未运行保留测试集，未冻结最终候选，未宣称效果提升。
- 未完成真实 Linux 整机 2 GiB RAM + 2 GiB swap 验收，不能以当前 Windows 内存数据替代。
- API 账单单价尚未核实，当前记录调用 token，不将未知费用计为零。
- B0 首轮启动后新增了脱敏异常链诊断和 HTTP 断连并发保护，未更改检索、提示词、模型或评分逻辑。开发集比较应注明这项代码哈希差异；测试集将在统一冻结代码下执行。

## 已有资源证据及适用范围

完整语料转换在 Windows 上用时 49.61 秒，应用进程峰值 RSS 212,918,272 bytes（约 203 MiB）。这是本地转换过程的测量，既不是整个部署内存，也不是最终 Linux 验收。

首批 320 个文本共 36,025 字符，embedding 消耗 27,408 token。按字符比例估计全量约 22,576,572 token，样本偏向首批表格、仅作量级估计。原始 float32 向量载荷约 771,104,768 bytes；Haystack 的 Python 对象和临时矩阵会额外占内存，baseline 获准超限。
