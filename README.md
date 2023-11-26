# LLMBasedAutonomousDriving
## 开源一份基于大模型的自动驾驶研究计划书，有意合作请联系
报名号：              
 
第一部分  课题简介
一、课题研究背景及主要内容
简要介绍研究问题或工程任务，包括国内外研究现状及发展预期情况。（不超过1000字）
课题名称：基于大型语言模型的多智能体自动驾驶决策算法研究
随着自动驾驶技术的快速进步，如何在多变的路况中做出准确和高效的决策成为了一个主要挑战。目前，自动驾驶系统在复杂路况中的能力仍有待提高，特别是在处理罕见或未预见事件时的适应性和决策能力方面。传统的自动驾驶系统，基于优化和模块化设计，面对复杂的长尾场景时性能受限，主要是因为它们缺乏类似人类的推理能力和经验知识。
近年来，大型语言模型（LLMs）在自然语言处理领域取得显著成就，展现出强大的推理、解释和记忆能力，为自动驾驶领域提供了新的思路。LLMs特别擅长处理分布外（OOD）推理、常识理解和知识检索，这些能力可以为自动驾驶系统提供类似人类的决策支持。因此，本研究的目标是探索和开发一种融合大型语言模型的自动驾驶决策算法，以提升系统在复杂路况下的适应性和智能化水平。
利用LLMs辅助自动驾驶已有一些研究。Chen等人引入了一种新颖的对象级多模态LLM架构，该架构结合了向量化的数值模态和预训练的LLM，以增强驾驶情境中的上下文理解。这项研究展示了LLM-driver在解释驾驶场景、回答问题和决策方面的熟练程度，突显了基于LLM的驾驶行为生成与传统行为克隆相比的潜力【1】。Fu等人探讨了使用LLM以类似人类的方式理解驾驶环境的可能性，并分析了LLM在处理复杂场景时的推理、解释和记忆能力。该研究确定了AD系统所需的三个关键能力：推理、解释和记忆，并通过构建闭环系统来展示LLM在驾驶场景中的理解和环境互动能力【2】。Sha等人研究了现有基于学习的AD系统在理解高级信息、泛化到罕见事件以及提供可解释性方面的挑战。该工作采用LLM作为复杂AD场景的决策组件，设计了认知路径，使LLM能够进行全面推理，并开发了将LLM决策转化为可执行驾驶命令的算法。这种方法使LLM的决策与PID控制器无缝集成，通过引导参数矩阵适应。这项研究是利用LLM作为复杂AD场景中有效决策者的初步尝试，从安全性、效率、泛化性和互操作性方面进行了考虑【3】。
【1】	Chen L, Sinavski O, Hünermann J, et al. Driving with llms: Fusing object-level vector modality for explainable autonomous driving[J]. arXiv preprint arXiv:2310.01957, 2023.
【2】	Fu D, Li X, Wen L, et al. Drive like a human: Rethinking autonomous driving with large language models[J]. arXiv preprint arXiv:2307.07162, 2023.
【3】	Sha H, Mu Y, Jiang Y, et al. Languagempc: Large language models as decision makers for autonomous driving[J]. arXiv preprint arXiv:2310.03026, 2023.
二、课题目标
课题拟达到的目标、成果的呈现形式或目标的评价方式等。（不超过500字）
本课题旨在开发和验证一种基于大型语言模型（LLMs）的多智能体自动驾驶决策算法，以提升自动驾驶系统在复杂路况下的决策能力和效率。具体目标如下：
2.1 构建和优化代码数据集：开发一个包含多种复杂路况（城市、高速、乡镇等）的大规模自动驾驶场景数据集。数据集将包括高质量的控制命令、环境感知数据、以及与之对应的语言描述，以支持多模态学习和算法验证。
2.2 开源代码和模型：为了促进学术界和工业界的进一步研究和应用，我们将开发的算法和模型以开源代码的形式发布。这将包括数据预处理、模型训练、评估和应用的完整代码。
2.3 多模态数据融合：研究和实现一种有效的多模态融合方法，将LLMs与自动驾驶系统中的视觉、雷达等感知数据结合起来，以提高模型对复杂驾驶环境的理解和适应能力。
2.4 开发多智能体自动驾驶决策算法：基于LLMs，设计和实现一种新型的多智能体自动驾驶决策算法。该算法将能够处理复杂的交通场景，包括城市拥堵、多车交互等情况，提供高效、准确的驾驶决策。
2.5 评估自动驾驶模型的效果：通过一系列的实验和模拟测试，评估所开发算法在各种路况下的表现。特别是在处理长尾事件和罕见情况时的效能，以及在多车辆交互和复杂交通环境中的表现。
三、课题拟采取的研究方法
课题研究方法或技术路线的可行性、先进性分析。（不超过500字）
课题拟将大型语言模型（LLM）作为自动驾驶智能体的一种模态，并基于多智能体强化学习（MARL）进行自动驾驶策略的训练。
3.1 多模态数据处理与LLM集成模块
此模块的目标是将视觉、雷达等多模态数据转换为可以被LLM理解和处理的格式，以便生成对自动驾驶有用的信息和建议。拟使用深度学习模型（如卷积神经网络CNN对于图像数据，循环神经网络RNN对于时序数据）来提取视觉和雷达数据的特征进行多模态数据预处理。这些特征将被LLM集成为文字信息，基于其对人类知识和语言的理解，LLM能够提供对场景的深入分析，如交通状况、潜在危险、环境变化等的描述和建议。以上的文本、图像、雷达等数据合并输入给自动驾驶智能体做驾驶策略输出。
3.2 多智能体强化学习（MARL）驾驶策略训练模块
该模块利用多智能体强化学习来训练自动驾驶策略，使智能体能够在复杂的交通环境中做出有效的驾驶决策。拟在仿真环境中部署多个基于LLM的强化学习智能体，通过多智能体间的相互作用，智能体能够学习如何复杂路况（如其他车辆、行人）的存在下执行驾驶决策。
该方案结合了LLM的高级语言理解能力和强化学习在复杂环境中的决策能力，为自动驾驶提供了一个创新的解决方案。通过该方法，可以期望智能体在处理复杂的道路情况和交互时表现出更高的智能决策能力和安全性。
四、课题主要预期创新点
不多于3个创新点，每项创新点的描述。（不超过300字）
4.1 创新点1：应用LLM Agents来增强自动驾驶智能体的决策能力。通过结合LLMs的高级语言处理和推理能力，我们旨在提升自动驾驶系统在处理复杂路况及与其他智能体（如其他车辆、行人）交互时的性能。这种方法将有助于自动驾驶系统更准确地理解和适应多变的交通环境。
4.2 创新点2：多模态数据融合策略。这种策略结合了自动驾驶系统的视觉、雷达等感知数据与LLMs的语言处理能力，以提高系统对复杂环境的理解和响应能力。这种融合方法有望提高自动驾驶系统在各种路况下的适应性和准确性。
4.3 创新点3：本课题计划开源其构建的大规模自动驾驶场景数据集和相关代码。这将为研究社区提供宝贵的资源，促进算法的进一步开发和优化。

五、预期成果的影响
课题的科学、技术、产业预期指标及社会、经济效益，在行业内的影响和效果等。（不超过500字）
5.1 科学与技术影响：通过本文提出的算法策略，本课题将提升自动驾驶系统在复杂路况（如城市、高速、乡镇等）的决策能力。LLM Agents的应用将为自动驾驶系统提供更高级的语言理解和推理能力、使自动驾驶车辆在处理与其他智能体交互时更加高效和安全，特别是在陌生或动态变化的复杂路况中。此外，开源的数据集和代码将促进自动驾驶技术的快速迭代和广泛应用，加速行业内的技术进步。
5.2 社会经济效益：本课题的研究成果预期将提高自动驾驶系统的安全性和可靠性，从而减少交通事故，提高道路使用效率。这不仅能够显著减少由交通事故引起的经济损失和社会成本，还能提高公众对自动驾驶技术的信任度，加速其社会接受度。此外，提高自动驾驶系统的决策能力和适应性，有望在减少交通拥堵、降低环境污染等方面产生积极效果。

 
第二部分  本人工作基础及进度安排
一、本人已有工作基础
申请人在所从事的科研及工程领域的前期或当前承担任务情况、相关研究成果及成效。（不超过500字）
本人在自动驾驶技术领域拥有丰富的研究和实践经验，前期主要工作集中在探索和实现基于大型语言模型（LLM）的端到端自动驾驶路径规划。通过基于Llama2-13B预训练的大型语言模型、并使用CLIP作为图像输入校准模型，该项工作展示了模型在零样本环境下的语义推理能力，这在自动驾驶领域是一项重要的突破。
与传统的模块化自动驾驶系统相比，该方法能够有效避免由于训练样本分布不均造成的长尾问题，并能对复杂和突发的交通情况进行更加精准的理解和响应。在此基础上，我们开发了Pegasus VLA框架用于整合多模态信息，这种VLA（视觉-语言自动驾驶）模型，通过将车辆的运行状态和轨迹规划转换为文本字符，并融入到对话训练集中，使得大型语言模型能够在训练过程中学习到实时场景理解和动态响应的能力。
实验结果表明，将VLA模型应用于自动驾驶场景中，能显著提升无人驾驶车辆在处理边缘问题时的性能，增强其对障碍物、静态目标及规划路径的泛化能力，展现出从网络级别数据中继承到的新兴功能。
二、所在单位相关科研及工程条件支撑状况
（不超过300字）
大模型算力资源：拥有XX块NVIDIA A100显卡，XX核心高性能计算集群。
经费资源：确保的研究经费总额为XXX万元。
师生资源：包括XX名自动驾驶和机器学习领域的教授，以及XX名研究生。
相关领域文章发表情况：已在国际期刊和会议上发表XXX篇相关论文。
三、课题相关的国内外合作与交流基础
（不超过300字）
国内外合作伙伴：
国内外高水平会议参与情况：
四、年度进展安排
计划实施的时间投入保障情况。（300~500字）
第一季度：项目启动与初步研究
文献调研，收集与分析当前大型语言模型（LLM）和自动驾驶领域的最新研究进展；
定义研究问题和假设，设计详细的技术路线；
第二季度：数据集准备与算法开发
收集、整理和预处理自动驾驶相关的多模态数据集，包括视觉、雷达等数据；
开发和测试多模态数据处理算法，确保数据能有效地融合并被LLM理解；
设计和实现多智能体强化学习（MARL）框架，开始初步的算法开发；
第三季度：算法优化与测试
在仿真环境中测试和调优基于LLM的强化学习智能体；
进行多智能体间的交互实验，优化智能体的协作和决策能力；
综合评估算法性能，对比基准模型，进行必要的调整和优化；
整理研究成果，开始撰写学术论文；
第四季度：论文撰写与成果发布
完成论文初稿，内部审阅和修改；
准备开源数据集和代码，撰写文档和使用指南；
提交论文至学术会议或期刊，准备项目总结报告和成果展示； 
第三部分  申请人承诺书
申请人承诺如下：
1.本人已完全理解招生简章的要求，并按简章要求准备该计划书；
2.本人对计划书、学历学位证书和所提交报名材料的真实性负责；
3.遵守《中华人民共和国保守国家秘密法》和《科学技术保密规定》等相关法律法规；
4.严守学术诚信。


如有不符，愿意承担相关后果并接受相应的处理。


                        申请人（签名）： 
2023年     月     日


![image](https://github.com/TimHoustons/LLMBasedAutonomousDriving/assets/107018384/d07f728e-9cc4-4fc9-8938-78beb9c617cc)