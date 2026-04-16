# Architectural Analysis of the SATAY-VIT Software Pipeline for Task-Aware Object Detection

The evolution of computer vision has reached a critical juncture where the mere identification of object categories is no longer sufficient for the requirements of high-level autonomous decision-making. Traditional object detection frameworks, while highly optimized for spatial localization, suffer from being task-agnostic, meaning they treat every instance of a detected class with identical priority regardless of the situational context. The SATAY-VIT proposal addresses the specific requirements of the DVCon India 2026 Design Contest by introducing a dual-pipeline software architecture designed to bridge the semantic gap through the fusion of high-speed spatial localization and contextual reasoning. This report provides an exhaustive analysis of the software pipeline logic, its alignment with the functional goals of task-driven detection, and its orchestration on the VEGA RISC-V processing ecosystem.

## Theoretical Foundation of the Semantic Gap and Affordance Perception

The conceptual motivation for SATAY-VIT is rooted in the "Semantic Gap Problem" (SGP), which identifies the mismatch between low-level machine-extractable features (pixels and bounding boxes) and the high-level semantic categories or functional utilities interpreted by humans. In a formal sense, if $V$ denotes the set of visual concepts and $L$ the set of lexical or semantic labels, the gap arises because the mapping $R \subseteq V \times L$ is often poorly specified or many-to-many, leading to inconsistent interpretations. Conventional detectors operate at the level of object labeling ($m_3$) but fail to achieve the level of semantic parsing and abstract reasoning ($m_4$) required for task-conditioned utility.

A primary manifestation of this gap in robotics is the inability to determine "affordance"—the potential action possibilities an object offers to an agent. Standard detectors might identify three different chairs in a room, but they cannot inherently distinguish which chair is "suitable for sitting comfortably" versus which is "stable enough to step on to reach a high shelf". This differentiation requires a nuanced understanding of an object’s individual appearance and its global context within the scene, a challenge that SATAY-VIT addresses by utilizing text semantics to guide spatial attention.

## Functional Logic of the 14 COCO-Tasks

The DVCon 2026 Problem Statement mandates the use of the COCO-Tasks dataset, which defines 14 functional goals that the system must satisfy. The software logic must move beyond simple category recognition to rank objects based on their fitness for a specific goal. The 14 tasks represent a wide spectrum of human daily activities where object selection is non-trivial and context-dependent.

| Task ID | Task Goal | Associated COCO Supercategories | Suitability Logic |
| :--- | :--- | :--- | :--- |
| 1 | Step on something | Furniture | Stability, surface area, height, material strength |
| 2 | Sit comfortably | Furniture | Padding, ergonomics, size, back support |
| 3 | Place flowers | Kitchen, Outdoor | Depth, water retention, stability, opening size |
| 4 | Get potatoes out of fire | Sports, Kitchen, Outdoor | Length, heat resistance, grip capability |
| 5 | Water plant | Kitchen, Indoor | Volume, pour mechanism, handle presence |
| 6 | Get lemon out of tea | Kitchen | Size of head, perforation, length |
| 7 | Dig hole | Sports, Kitchen, Indoor | Structural rigidity, edge sharpness, leverage |
| 8 | Open bottle of beer | Furniture, Kitchen, Indoor | Leverage points, hardness, grip |
| 9 | Open parcel | Kitchen, Indoor | Sharpness, point of entry, handle control |
| 10 | Serve wine | Kitchen | Material (glass vs. ceramic), shape, stem presence |
| 11 | Pour sugar | Kitchen | Capacity, flow control, spill-prevention shape |
| 12 | Smear butter | Kitchen | Surface area, flatness, flexibility |
| 13 | Extinguish fire | Kitchen, Indoor | Volume, liquid containment, non-flammability |
| 14 | Pound carpet | Sports | Surface area, weight, handle length |

The logical requirement for "Task 10: Serve wine" serves as a benchmark for the SATAY-VIT architecture. If both a wine glass and a standard cup are present, the system should prioritize the wine glass. However, if no wine glass is available, the system must adapt its behavior to prioritize the cup as the next best alternative, rather than failing to return a result. This adaptive behavior is a core requirement of the design contest and is facilitated by the "Rank-and-Filter" software logic.

## The YOLO Localizer: Spatial Logic and Streaming Principles

The first functional engine in the SATAY-VIT dual-pipeline is the YOLO "Localizer," utilizing the YOLOv11n backbone. The logical goal of this component is high-speed spatial detection, providing the "where" and "what" for all objects in a $640 \times 640$ RGB pixel stream. YOLOv11n is selected for its architectural efficiency, featuring a Cross-Stage Partial (CSP) bottleneck and C3k2 blocks that maximize gradient flow while maintaining a low parameter count of approximately 2.6 million.

The Localizer logic is governed by the SATAY (Streaming Architecture Toolflow for YOLO) framework, which addresses the "Memory Wall" problem inherent in edge devices with limited on-chip memory. The software pipeline for the Localizer is built on three fundamental logical principles:

### Pipeline Parallelism and Layer Streaming

Instead of a conventional "Load-then-Execute" model where an entire network is brought into memory before computation, the Localizer logic treats the FPGA fabric as a purely pipelined processor. Every layer of the network is implemented as an active hardware stage, and data moves through these stages in a streaming fashion. This allows downstream layers to begin their specific mathematical operations as soon as partial outputs from upstream layers are generated, significantly reducing total inference latency.

### Weight-Stationary Execution and Ping-Pong Buffering

To operate within the 2.0 MB Block RAM (BRAM) limit of the Genesys 2 Kintex-7 board, the Localizer logic adopts a weight-stationary, layer-pipelined approach. The software manages parameters as transient residents of fast memory; at any given moment, only the weights for the currently active and immediately subsequent layers are stored on-chip. The VEGA core orchestrates a Ping-Pong Buffer logic: while the systolic matrix-multiply array is processing Layer $N$ using weights in Bank A, the AXI-DMA is concurrently prefetching weights for Layer $N+1$ from off-chip DDR3 memory into Bank B. This strategy reduces the peak memory footprint from a theoretical 2.6 MB (for YOLO alone) to a manageable 354 KB.

### Sliding Window and Feature Stream Logic

The spatial detection logic does not require the storage of a complete $640 \times 640$ frame. Instead, it utilizes line buffers to store only three rows of the incoming feature map at a time. As the fourth row arrives, the first is discarded, maintaining a sliding window generator that feeds the matrix-multiply engine. This logic is highly efficient, requiring only $(K-1) \times W \times C$ words of on-chip storage, where $K$ is the kernel size, $W$ is the feature map width, and $C$ is the number of channels.

## The ViT Reasoner: Semantic Alignment Logic

The second functional engine is the ViT "Reasoner," based on the Memory-Efficient Vision Transformer (ME-VIT) architecture. While the YOLO engine focuses on local features, the Reasoner logic provides the global context and semantic justification required to distinguish between functionally distinct objects of the same class.

### Sparse Attention Mechanism

The computational cost of traditional Vision Transformers is dominated by the Multi-Head Self-Attention (MHSA) module, which typically involves all-to-all patch communication. To meet the edge-AI requirements of the DVCon contest, the Reasoner logic employs a sparse attention mechanism. Instead of calculating attention scores across every possible patch combination, the software logic focuses exclusively on the top-$N$ most relevant regions relative to the task prompt. This approach reduces memory traffic and activation buffering requirements by approximately 8x, enabling parallel execution alongside the YOLO pipeline on the same FPGA fabric.

### Cross-Attention and Task Embeddings

The reasoning process begins with the integration of text semantics. The system takes a text prompt corresponding to one of the 14 COCO-Tasks and selects a pre-stored 512-dimensional (512D) task embedding. These embeddings are high-level mathematical representations that capture the functional requirements of a task (e.g., "cutting" or "drinking").

The ME-VIT Reasoner logic then transforms the $640 \times 640$ input frame into high-level mathematical vectors through a patch embedder. A cross-attention head then compares these "image vectors" against the "task vector". The logic calculates relevance scores that signify how well a specific region of the image satisfies the functional requirements of the task.

### Heatmap Generation and Logical Justification

The output of the Reasoner is a $16 \times 16$ patch-relevance heatmap. In this heatmap, regions of high relevance are assigned higher scores (e.g., 0.95), while irrelevant regions receive low scores (e.g., 0.10). This software logic provides an inherent interpretability layer—a mechanism that logically justifies "why" a particular object was chosen. For instance, in a "cutting" task, the cross-attention scores will be highest for the "blade" portion of a knife, rather than the handle or the table surface, providing a mathematical evidence trail for the final decision.

## VEGA Core: Post-Processing and Task Fusion Orchestration

The VEGA AS1061 is an indigenously developed 64-bit RISC-V core (RV64IMAFDC) that serves as the "Brain" or system orchestration hub for the SATAY-VIT architecture. Its primary responsibility is to manage high-level, non-deterministic, and non-linear logic that would be inefficient if implemented directly in FPGA Look-Up Tables (LUTs).

### HW/SW Partitioning Strategy

To maintain a frame rate of 25 FPS while ensuring reliable timing closure, SATAY-VIT employs a rigorous hardware-software co-design. The FPGA fabric handles the parallel arithmetic of convolutions and attention blocks, while the VEGA core manages sequential post-processing tasks.

| Logic Category | Operational Task | Location |
| :--- | :--- | :--- |
| Parallel Arithmetic | Convolutions (3x3, 1x1), Matrix-Vector Multiply | FPGA Fabric (Systolic Array) |
| Parallel Dataflow | Line Buffering, Feature Streaming, Weight Prefetching | FPGA Fabric (AXI-Stream) |
| Sequential Filtering | Non-Maximum Suppression (NMS) | VEGA RISC-V Core |
| Non-Linear Logic | Softmax Ranking, Score Fusion, BBox Alignment | VEGA RISC-V Core |
| System Control | Ping-Pong Buffer management, FreeRTOS Interrupts | VEGA RISC-V Core |

### Non-Maximum Suppression (NMS) Software Logic

The YOLO Localizer produces multiple candidate bounding boxes for the same object instance. The VEGA core executes the NMS algorithm to filter these redundant detections. The software logic calculates the Intersection over Union (IoU) between overlapping boxes and suppresses those with lower confidence scores, ensuring that only the most spatially accurate candidates are passed to the fusion stage.

### Softmax Approximation and Ranking

To save BRAM resources and improve latency, the software logic implements an "exponent-less Softmax". Traditional Softmax calculations require transcendental operations that are computationally expensive for edge processors. SATAY-VIT replaces these with piecewise linear or power-of-two approximations using bit-shifts. This logic allows the system to normalize the $16 \times 16$ cross-attention scores into a probability-like space with minimal overhead.

## Mathematical Modeling of Task Fusion

The "Task Fusion" stage is the final step in the SATAY-VIT software pipeline where the outputs of the "Localizer" and "Reasoner" are merged to produce a task-aware result. This logic resolves the "Semantic Gap" by aligning spatial localization data with semantic relevance scores.

### Score Fusion Algorithm

For every bounding box $B_i$ generated by the YOLO engine, the Localizer provides a spatial confidence score $C_i \in [0, 1]$. Simultaneously, the ME-VIT Reasoner provides a patch-relevance heatmap $H$, where each patch $p_{x,y}$ has a relevance value $R_{x,y} \in [0, 1]$.

The fusion logic on the VEGA core performs the following steps:

1. **Coordinate Mapping**: The spatial coordinates of bounding box $B_i$ are mapped onto the $16 \times 16$ grid of the heatmap $H$.
2. **Relevance Integration**: The software calculates the average relevance score for the patches contained within or overlapping with the bounding box area:
   $$R_{box, i} = \frac{1}{N_{patches}} \sum_{(x,y) \in B_i} R_{x,y}$$
3. **Final Task-Aware Scoring**: The final score for object $i$ is calculated by multiplying its spatial confidence by its semantic relevance:
   $$S_{final, i} = C_i \times R_{box, i}$$
4. **Argmax Selection**: The VEGA core identifies the object with the highest $S_{final}$ as the "most appropriate object" for the given task input.

### Justification Logic in Decision Making

This scoring mechanism ensures that the system satisfies the hierarchical preference requirements of the COCO-Tasks dataset. In the wine glass example, if the system detects a wine glass ($C_{wine} = 0.95$) and a ceramic cup ($C_{cup} = 0.98$), their relative spatial confidences are nearly identical. However, for the task of "serving wine," the Reasoner will assign the wine glass a higher relevance ($R_{wine} = 0.90$) compared to the cup ($R_{cup} = 0.40$). The final score fusion will prioritize the wine glass ($S_{wine} = 0.855$ vs. $S_{cup} = 0.392$). If the wine glass is removed, the $S_{cup}$ remains the highest available score, allowing the system to pivot intelligently.

## Addressing Contest Requirements: Novelty and Efficiency

The SATAY-VIT architecture is specifically engineered to meet the evaluation criteria outlined in the DVCon India 2026 Problem Statement.

### Novelty and Accuracy

The primary novelty of the proposed pipeline is the dual-engine parallel processing model that fuses spatial and semantic data in software on a RISC-V core. Unlike standard one-stage or two-stage detectors that prioritize only mAP (Mean Average Precision) on fixed labels, SATAY-VIT optimizes for "Decision Relevance"—the ability to identify the most appropriate object based on functional goal descriptions. The use of lightweight language models (represented by the 512D text embeddings) allows for a dynamic, prompt-driven filtering process that is absent in conventional frameworks.

### Inference Latency and Determinism

The contest emphasizes inference latency as a key metric. SATAY-VIT achieves a consistent 25 FPS throughput (40 ms per frame) by utilizing the Streaming Architecture philosophy. The logic ensures that image I/O, convolution processing, and semantic reasoning occur in parallel, meaning the hardware pipeline never stalls while waiting for data. The VEGA core, running FreeRTOS, provides deterministic interrupt handling for the Task Fusion stage, ensuring that software overhead remains below 5 ms per frame.

### Resource and Power Optimization

The architectural decision to offload sequential logic to the VEGA core preserves the Kintex-7 FPGA's Look-Up Tables (LUTs) for parallel arithmetic. The proposal estimates that this HW/SW partitioning will keep LUT utilization at a safe 52% and peak BRAM utilization below 10%. By minimizing the movement of weights through "Temporal Weight Folding" and reducing attention traffic via sparse mechanisms, the system significantly lowers power and energy consumption compared to standard GPU or desktop CPU implementations.

| Performance Metric | Target Specification | Logic/Mechanism Used |
| :--- | :--- | :--- |
| Frame Rate | 25 FPS (40 ms latency) | Deep Pipelining & Streaming |
| Weight Footprint | 354 KB Peak BRAM | Temporal Weight Folding & Ping-Ponging |
| Task Logic | 14 functional tasks | 512D Task Embeddings & Cross-Attention |
| CPU Overhead | < 5 ms per frame | FreeRTOS & AXI-Stream Orchestration |
| Power Budget | < 15W Estimated | Sparse Attention & Single-Load Weights |

## Conclusion: A Paradigm for Task-Conditioned Edge Perception

The SATAY-VIT software pipeline architecture represents a mathematically grounded and physically viable solution to the task-aware object detection requirements of the DVCon India 2026 challenge. By synthesizing high-speed spatial identification with memory-efficient semantic reasoning, the system successfully bridges the "Semantic Gap" that has historically limited the utility of object detectors in autonomous and assistive robotics.

The architecture’s strength lies in its dual-pipeline parallel execution and its sophisticated hardware-software partitioning on the VEGA RISC-V core. The YOLO Localizer provides the spatial foundation, while the ME-VIT Reasoner supplies the semantic justification, with the final Rank-and-Filter decision logic managed by the VEGA core. This integrated approach ensures that the system can accurately prioritize objects based on functional suitability while maintaining the real-time throughput and low power consumption required for modern edge-AI deployments. Through its innovative weight management and score fusion logic, SATAY-VIT fulfills the core mandate of the contest: creating a detection framework that does not just see objects, but understands their purpose in the context of human goals.