# 验证策略

## 1. 结论先行

v0.1 的核心验证必须在单机多卡上跑：至少 2 张支持 CUDA P2P 的 GPU，多个进程，一 rank 一 GPU。

- CPU 验证算法参考、切分和控制面，不会证明 CUDA kernel、P2P 或 GPU 内存序正确。
- 单卡验证构建、API、stream 和 `nranks=1` identity 路径，不会证明 ring 正确。
- 单机多卡才验证 v0.1 的 CUDA IPC/P2P Ring 与 FIFO 闭环。
- 多节点多卡只属于 v0.2 的 Socket proxy 验证。

因此不能在 CPU 与单卡之间二选一。采用分层验证，每层只承担它能证明的结论；v0.1 的最终门槛是多卡。

## 2. 当前环境限制

初始实现环境记录为没有 `cmake`、`nvcc` 和可用 GPU。本轮已用 Apple Clang 17 以 `-std=c++17 -Wall -Wextra -Wpedantic -Werror` 直接编译并运行纯 CPU Ring/ticket 模型，结果通过；同时完成了源码静态审查。没有编译或运行 CUDA 库，因此不能声称：

- CUDA 源码已成功编译；
- kernel 已启动；
- CUDA IPC handle 可跨进程打开；
- peer load/store 与 system-scope 同步成立；
- 多 rank Ring 无死锁；
- 结果或性能已在 GPU 上验证。

这些结论必须在 CUDA 主机补验，并记录 GPU 型号、driver/runtime 版本和 P2P matrix。

下面主要是验收计划，不是已通过清单。当前只有 CPU Ring/ticket 模型实际通过；GPU smoke example 仅有源码，bootstrap 测试、完整 API 负向矩阵、单卡和多卡验收仍待实现或运行。

## 3. 分层矩阵

| 层级 | 环境 | 验证对象 | 能证明 | 不能证明 | 版本门槛 |
|---|---|---|---|---|---|
| L0 静态 | 当前 CPU 环境 | API、边界检查、代码结构、协议不变量 | 设计一致性与明显错误 | 编译和运行时行为 | 提交前必做 |
| L1 CPU | 任意主机 | segment/tile 计算、Ring 索引参考、bootstrap framing | 纯函数和 socket 控制面逻辑 | CUDA/P2P/FIFO 内存序 | v0.1 辅助 |
| L2 单卡 | 1 GPU | CUDA 构建、`nranks=1`、copy/in-place、stream/error | CUDA 基础集成 | rank 间通信与死锁 | v0.1 必做但不充分 |
| L3 单机多卡 | 2+ P2P GPU | 多进程 IPC/P2P Ring、FIFO、连续操作 | v0.1 核心闭环 | 跨节点 proxy | v0.1 验收门槛 |
| L4 多节点 | 2+ nodes/GPUs | TCP framing、proxy、背压与失败传播 | v0.2 核心闭环 | RDMA/生产性能 | v0.2 验收门槛 |

## 4. 统一正确性 oracle

rank `r` 的输入使用确定性、可重现且不易相互抵消的数据，例如：

$$
x_r[i] = r \times 0.25 + (i \bmod 97) \times 0.001
$$

CPU reference 为：

$$
y[i] = \sum_{r=0}^{P-1} x_r[i]
$$

每个 rank 将 device 结果复制回 host 后与同一 reference 比较。FP32 加法顺序不同会产生舍入差异，因此使用绝对误差和相对误差联合判断：

$$
|a-b| \le \epsilon_{abs} + \epsilon_{rel}|b|
$$

容差按 rank 数与输入幅度设定并在测试中明确，不用 bitwise equality。测试失败必须报告 rank、index、expected、actual、absolute/relative error。

## 5. L0/L1：CPU 与静态验证

这些测试应完全不依赖 GPU：

### 5.1 切分性质

- `count=0`，仍生成一个零 payload 控制 tile，且不访问 tensor 地址。
- `count < nranks`，产生空 segment。
- `count == nranks`。
- `count` 不能被 `nranks` 整除。
- segment 正好一个 tile、跨两个 tile、尾 tile 只有一个元素。
- 所有 segment 覆盖 `[0,count)`，互不重叠，长度差最多 1。

### 5.2 Ring 索引模型

用 CPU reference 模拟 Ring，检查：

- Reduce-Scatter 后每个 segment 只由一个 owner 持有，值等于所有 rank 对应输入之和。
- All-Gather 后每个 rank 拥有全部 reduced segments。
- 每个逻辑消息 `(operation, phase, step, tile)` 只有一个 producer 和一个 consumer。
- 所有 rank 的消息数量、ticket 区间和步骤次序一致。

当前 CPU model 已与 device kernel 统一为 `rank r` 首发 segment `r-1`、Reduce-Scatter 后持有 segment `r` 的旋转。它验证最终算法和基本 slot/ticket 复用；但尚未逐 edge 比较 device 的 segment/ticket trace。若增加这种检查，应复用 CUDA-free 的纯索引函数，但不能复用 GPU 搬运实现，否则相同 bug 可能同时存在于被测代码与 oracle。

### 5.3 bootstrap

- unique id 编解码和非法长度。
- 多进程 rendezvous 的 rank 去重、缺失和越界。
- partial read/write、peer 提前关闭、超时。
- bootstrap 阶段某 rank 报错时，其余 rank 通过状态传播、连接关闭或 bootstrap timeout 有界退出；这不适用于 steady-state GPU wait。

## 6. L2：单卡验证

单卡运行 `nranks=1`：

- out-of-place 等价于 device-to-device identity copy。
- in-place 不修改数据。
- `count=0`、小 count、大 count、非默认 stream；`count=0` 仍应安全推进控制状态。
- API 返回后 stream 未被内部全局同步；用 event 验证顺序。
- 错误 device、空指针、重叠但非完全 in-place buffer、切换 stream 被拒绝。
- communicator 重复创建/销毁，无 CUDA 资源泄漏。

单卡通过只说明基础集成可用，不能把它标记为 v0.1 Ring 通过。

## 7. L3：v0.1 单机多卡验收

### 7.1 环境前置检查

- 至少 2 张 GPU；建议再用 4 张 GPU 覆盖更长 ring。
- 每个相邻 rank 的 `cudaDeviceCanAccessPeer` 满足要求。
- 若同步协议依赖 native peer atomics，查询相应 P2P attribute 并要求支持。
- 记录 rank -> device 映射与 `nvidia-smi topo -m` 或等价 P2P matrix。

不满足时测试应明确 skip/unsupported，不能改走 host copy 后宣称 P2P 通过。

### 7.2 正确性用例

对 `P=2` 和可用时的 `P=4`，覆盖：

- out-of-place 与完全 in-place。
- `count` 集合：`0`、`1`、`P-1`、`P`、`P+1`、tile 边界前后、多个 tile、数十 MB。
- 非默认 stream，并用 CUDA event 建立输入生产 -> AllReduce -> 输出消费的依赖。
- 连续数百次 collective，输入每次改变，捕获 ticket/flag 复用和 ABA 问题。
- 让 count 交替跨过 tile 边界，验证不同 operation 的 ticket 区间不会重叠。
- 在 `count=0` 与非零 count 之间反复切换，验证零 payload 控制 tile 后下一次 operation 不死锁。
- 每个 rank 都与同一 CPU oracle 比较。

### 7.3 并发与负向用例

- 各 rank 的 `count` 或 collective 次序不一致属于契约违反；v0.1 没有 GPU-visible abort，kernel 可能持续等待。
- 某 rank 不参与或提前退出时，bootstrap 阶段自身有超时；steady-state GPU wait 可能不返回，由测试 harness 的外部 timeout 强制回收全部进程。
- 同一 communicator 换 stream、换 current device、并发 host thread：返回 unsupported/invalid。
- 不可 P2P 的 rank/device 映射：初始化明确失败。

### 7.4 FIFO 与有界性

- 输入从小 tensor 扩大到远大于 mailbox，记录通信分配保持不变。
- 使用多 tile 和大量连续 operation 迫使每个物理 slot 多次复用。
- debug build 可记录每 slot 的 ticket 迁移，检查未获得 credit 不覆盖、未看到 ready 不读取。
- 测试设定进程级超时并保留最后 operation/phase/step/ticket，便于定位死锁。

### 7.5 工具

在支持环境中按可用性使用：

- Compute Sanitizer `memcheck`：越界和非法地址。
- Compute Sanitizer `racecheck`：适用范围内的数据竞争；跨 GPU 可见性仍需协议审查和压力测试。
- AddressSanitizer/UndefinedBehaviorSanitizer：bootstrap 与 host 生命周期。
- 外部进程级 timeout：任何 collective 测试都不能无限占住 CI；它是 harness 兜底，不代表库提供 runtime abort。

工具通过不是内存序证明；内存序必须同时由 CUDA 文档允许的 scope/order 和代码不变量支撑。

## 8. L4：v0.2 多节点验收

除重复 L3 正确性矩阵外，至少验证：

- 两个节点各一 rank/GPU，ring 有真实跨节点 Socket 边。
- 本机边走 P2P、远端边走 proxy，消息 sequence 连续一致。
- 人为缩小 Socket send/recv buffer，稳定触发 partial I/O 和背压。
- 限制 proxy 进度或注入延迟，host/GPU FIFO 仍有界。
- 对端关闭、header 损坏、payload 长度越界时，相关 proxy/control path 有限时间检测；若 GPU 已在 flag 上等待，由外部 timeout 回收进程。
- destroy 与错误路径不会留下 proxy thread、socket 或 pinned memory。

不以带宽数字作为 v0.2 完成条件；只记录吞吐和延迟用于发现数量级回退。v0.2 的目标是解释 proxy 闭环，不是与 NCCL/RDMA 竞速。

## 9. 验收记录模板

每次声称版本通过时保留：

```text
commit:
host / OS:
CUDA driver / runtime / nvcc:
GPU model and count:
rank -> device mapping:
P2P capability matrix:
build type and flags:
test command:
passed / skipped / failed cases:
sanitizer command and result:
known limitations:
```

如果某层因硬件不可用而未运行，状态写“未验证”，而不是“预期通过”。截至本文当前记录，CPU Ring/ticket 模型已由 Apple Clang 17 以严格告警编译并通过；bootstrap CPU 测试、L2 单卡、L3 单机多卡和 L4 多节点均未运行。CUDA v0.1 只有静态审查，v0.2 只有设计。
