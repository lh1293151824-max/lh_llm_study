# ==================== 本次任务 ====================
# 任务名称：增加 DataParallel 多 GPU 训练与 Hugging Face 参数别名兼容
# 用户已确认的任务：第 22 项，创建临时任务文件并写入具体实施伪代码
# 项目根目录：当前打开的项目根目录
# 参考代码：
# https://github.com/datawhalechina/happy-llm/tree/main/docs/chapter5/code
# 仅参考 ddp_pretrain.py、ddp_sft_full.py、k_model.py 中的 DataParallel
# 基础用法以及 forward 对 input_ids、labels 的兼容方式，不整体复制代码。

# ==================== 本次任务目标 ====================
# 1. 保留 pretrain 和 SFT 共用的 train.py 训练入口。
# 2. 通过 torch.nn.DataParallel 增加可配置的基础多 GPU 训练。
# 3. 单卡 CUDA 和 CPU 环境继续使用当前训练逻辑，不强制启用 DataParallel。
# 4. Transformer.forward() 同时兼容现有 idx/targets 和 HF input_ids/labels。
# 5. 保留当前字典返回值、loss mask、左 padding、attention mask 和 DeepONet 输出头逻辑。
# 6. 保留当前完整 checkpoint 格式、严格权重加载和 tokenizer 兼容检查。
# 7. DataParallel checkpoint 始终保存底层模型权重，不写入 module. 前缀。

# ==================== 预计修改文件 ====================
# 1. config.py
# 2. train.py
# 3. k_model.py
# 不创建新的业务模块，不修改 dataset.py、model_sample.py、export_model.py、deeponet.py。

# ==================== config.py 修改伪代码 ====================
# 修改目的：集中配置 DataParallel，默认行为在 GPU 数量不足时安全回退。
#
# 在 Runtime and training configs 附近新增：
# USE_DATA_PARALLEL = True
# DATA_PARALLEL_DEVICE_IDS = None
#
# 参数语义：
# USE_DATA_PARALLEL=False：始终不包装模型。
# USE_DATA_PARALLEL=True 且 CUDA 可用 GPU 数量大于 1：启用 DataParallel。
# USE_DATA_PARALLEL=True 但只有一张 GPU 或没有 CUDA：记录回退信息，继续单卡或 CPU。
# DATA_PARALLEL_DEVICE_IDS=None：使用所有当前可见 CUDA 设备，编号为 0..device_count-1。
# DATA_PARALLEL_DEVICE_IDS=[0, 1]：只使用指定的当前可见设备。
# 设备编号不合法、重复、为空或超出 torch.cuda.device_count() 时直接报错。
# 不在代码运行过程中修改 CUDA_VISIBLE_DEVICES，避免 CUDA 初始化顺序产生歧义。

# ==================== train.py 修改伪代码 ====================
# 修改目的：让训练初始化、验证、保存和恢复都正确适配 DataParallel。

# 一、build_args_from_config()
# args.use_data_parallel = default_config.USE_DATA_PARALLEL
# args.data_parallel_device_ids = default_config.DATA_PARALLEL_DEVICE_IDS

# 二、模型解包工具
# def unwrap_model(model):
#     if isinstance(model, torch.nn.DataParallel):
#         return model.module
#     return model
#
# 所有需要访问模型结构、config 或原始 state_dict 的位置统一调用 unwrap_model()。

# 三、设备配置解析
# def resolve_training_device(args):
#     if not torch.cuda.is_available():
#         args.resolved_data_parallel_device_ids = []
#         return torch.device("cpu")
#
#     available_count = torch.cuda.device_count()
#     if args.data_parallel_device_ids is None:
#         device_ids = list(range(available_count))
#     else:
#         验证配置必须是非空的整数序列
#         验证设备编号非负、无重复并且小于 available_count
#         device_ids = list(args.data_parallel_device_ids)
#
#     args.resolved_data_parallel_device_ids = device_ids
#     return torch.device(f"cuda:{device_ids[0]}")
#
# prepare_training() 使用 resolve_training_device(args)，并记录：
# use_data_parallel
# requested/resolved device IDs
# available CUDA device count
# primary device

# 四、DataParallel 包装
# def maybe_wrap_data_parallel(model, args, device, log_file):
#     if device.type != "cuda" or not args.use_data_parallel:
#         return model
#     device_ids = args.resolved_data_parallel_device_ids
#     if len(device_ids) <= 1:
#         记录未启用多卡以及回退原因
#         return model
#     确认模型已经位于 cuda:device_ids[0]
#     model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
#     记录实际启用的 GPU 数量与编号
#     return model

# 五、init_model() 初始化顺序
# base_model = Transformer(config=model_config).to(device)
# 使用 base_model 统计参数量
# 如果是 SFT 初始化：在包装前把预训练权重严格加载到 base_model
# optimizer = AdamW(base_model.parameters(), lr=args.learning_rate)
# scaler = GradScaler(...)
# 如果 resume：在包装前恢复 base_model、optimizer、scaler
# model = maybe_wrap_data_parallel(base_model, args, device, log_file)
# model.train()
# 返回 model、optimizer、scaler 和现有数据加载器等返回值
#
# 说明：DataParallel 包装不会复制出新的 Parameter 对象供 optimizer 管理，
# 因此在包装前创建的 optimizer 仍然管理底层模型参数。

# 六、checkpoint 保存
# def _checkpoint_model_config(model):
#     base_model = unwrap_model(model)
#     config = base_model.config
#     保持现有 model_config 字段不变
#
# def save_checkpoint(...):
#     base_model = unwrap_model(model)
#     checkpoint["model_state_dict"] = base_model.state_dict()
#     继续保存 optimizer_state_dict、scaler_state_dict、model_config、
#     tokenizer_config、train_stage、epoch、global_step 和 avg_loss
#
# 这样单卡与多卡 checkpoint 的模型键完全一致，不产生 module. 前缀，
# model_sample.py、export_model.py、SFT 初始化和 resume 继续严格加载。

# 七、验证逻辑
# def evaluate_loss(...):
#     训练/验证模式切换继续作用于外层 model
#     outputs = model(x, labels, attention_mask=attention_mask)
#     保留现有 last_loss 与 loss_mask 计算
#     scale_ratio = outputs.get("output_scale_ratio")
#     如果 DataParallel 把每张卡的标量汇总为一维张量：
#         ratio_value = scale_ratio.detach().max().item()
#     否则同样通过 max().item() 得到标量
#     warning 阈值读取 unwrap_model(model).config.operator_scale_warning_ratio
#
# DataParallel 按 batch 维拆分 x、labels、loss_mask 和 attention_mask。
# 模型返回的逐 token last_loss 会按设备顺序在 dim=0 汇总；现有 loss_mask.view(-1)
# 与 batch 顺序保持一致，因此不改动当前 mask 加权损失公式。

# 八、训练循环
# 保留 outputs = model(x, labels, attention_mask=attention_mask)
# 保留 outputs["last_loss"]、梯度累积、AMP、梯度裁剪、验证和保存周期。
# torch.nn.utils.clip_grad_norm_(model.parameters(), ...) 可继续用于包装模型。
# 不引入 DistributedSampler，不把本任务扩展为 DDP。

# 九、日志
# 记录 DataParallel 是否实际启用、主设备、设备编号和 GPU 数量。
# 参数量统计只统计底层模型一次，不按 GPU 副本数量重复计算。

# ==================== k_model.py 修改伪代码 ====================
# 修改目的：兼容 Hugging Face 调用参数，同时保持项目旧调用完全可用。

# Transformer.forward() 签名调整为：
# def forward(
#     self,
#     idx=None,
#     targets=None,
#     attention_mask=None,
#     input_ids=None,
#     labels=None,
# ):
#
# 参数解析：
# if idx is not None and input_ids is not None:
#     raise ValueError("pass only one of idx or input_ids")
# if targets is not None and labels is not None:
#     raise ValueError("pass only one of targets or labels")
# if idx is None:
#     idx = input_ids
# if targets is None:
#     targets = labels
# if idx is None:
#     raise ValueError("idx or input_ids is required")
#
# 参数映射完成后，继续执行当前所有检查和计算：
# idx/input_ids 必须为二维整数张量
# targets/labels 必须与输入形状和设备一致
# attention_mask 必须与输入形状一致并转换为 bool
# 训练时输出全序列 logits 和逐 token last_loss
# 推理时只对最后位置计算 logits
# 返回 self.OUT，保留 logits、last_loss 和可选 output_scale_ratio
#
# 以下调用全部兼容：
# model(x)
# model(x, labels)
# model(x, labels, attention_mask=mask)
# model(idx=x, targets=labels, attention_mask=mask)
# model(input_ids=x, labels=labels, attention_mask=mask)
#
# generate() 不修改，继续使用现有 self(idx_cond, attention_mask=mask_cond)。
# 不增加未确认的 position_ids、token_type_ids、past_key_values、return_dict 等参数。

# ==================== checkpoint 兼容规则 ====================
# 1. 新旧单卡 checkpoint：继续严格加载。
# 2. 新 DataParallel checkpoint：保存底层模型权重，格式与单卡 checkpoint 相同。
# 3. 已带 _orig_mod. 前缀的历史权重：继续走 normalize_state_dict_keys()。
# 4. 不新增 module. 自动清理作为常规路径；新保存逻辑保证不生成该前缀。
# 5. optimizer 和 scaler 状态保持当前格式，resume 语义不变。
# 6. model_config、tokenizer_config、DeepONet 配置和训练阶段字段保持不变。
# 7. 不降低 strict=True 权重加载要求。

# ==================== 预计静态检查 ====================
# 只读命令：
# rg -n "DataParallel|unwrap_model|model.config|model.state_dict|input_ids|labels" config.py train.py k_model.py
# rg -n "strict=False|module\\.|CUDA_VISIBLE_DEVICES" --glob "*.py" .
# git diff --check
# git diff -- config.py train.py k_model.py
#
# 检查目标：
# 1. 所有 DataParallel 下的 config/state_dict 访问都经过解包。
# 2. 保存的 checkpoint 不产生 module. 前缀。
# 3. HF 参数别名不会与旧参数静默冲突。
# 4. loss mask 和 attention mask 调用方式不变。
# 5. generate() 没有修改。
# 6. model_sample.py 和 export_model.py 不需要适配 DataParallel 包装。

# ==================== 代码运行计划 ====================
# 本阶段不运行 Python、训练、推理或数据处理代码。
# 业务代码修改后也先执行上述只读静态检查。
# 如需进行单卡或多卡动态测试，必须另行确认运行命令、工作目录和预期输出。
# 当前环境不是 Windows，无法按 AGENTS.md 要求使用 Start-Process 弹出 PowerShell，
# 因此在规则未被用户明确调整前不执行动态测试。

# ==================== 依赖与环境 ====================
# 是否安装依赖：否，torch.nn.DataParallel 已属于现有 PyTorch。
# 是否修改依赖配置文件：否。
# 是否影响当前运行环境：否。
# 是否访问项目根目录外文件：不访问本地项目外文件。
# 参考的 GitHub 文件由用户明确指定，仅用于只读设计参考。
# 是否执行 Git 写操作：否。

# ==================== 需要讨论与不确定的地方 ====================
# 1. 推荐 USE_DATA_PARALLEL=True、DATA_PARALLEL_DEVICE_IDS=None，自动使用所有可见 GPU；
#    GPU 数量不足时自动回退而不是报错。
# 2. 当前方案不设置 CUDA_VISIBLE_DEVICES；如需屏蔽物理 GPU，应在启动进程前由用户设置环境变量。
# 3. 当前方案是单进程 DataParallel，不包含 DDP、DistributedSampler 或多进程启动器。
# 4. 除上述三点外，暂无需要讨论或确认的事项。
