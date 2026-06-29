# The Acceleration Curve
### *Where the evidence already points for CORDIC across the AI silicon stack, 2026–2035*

---

> This document makes twelve grounded predictions about where CORDIC-based computation goes in the AI hardware stack over the next decade. Each prediction is built from first principles, grounded in published 2025–2026 silicon results, and held to a standard of falsifiability: each one names specific conditions under which it would be wrong. Confidence levels are explicit. Timelines are specific. The goal is not to be right about everything — it is to be honest about what the evidence already implies and to be wrong in the most useful possible way.

---

## Why Silicon Predictions Are Different — and Why These Ones Have Standing

Forecasting software is hard. Forecasting silicon is almost impossible.

Hardware has a three-to-five-year development cycle from architecture decision to production chip. The people making those decisions are working from today's data to build products that will compete in markets that do not yet exist. An AI chip that tapes out in 2026 ships to customers in 2028 and defines someone's infrastructure for five to seven years after that. The stakes are extreme, the feedback loops are slow, and the decisions are irreversible in a way that software is not.

And yet silicon is also, paradoxically, the most predictable domain in technology. Not because engineers are clairvoyant, but because physics does not negotiate. Power density, transistor scaling, memory bandwidth, clock speed — these are not subject to pivots or funding cycles or shifts in fashion. When the evidence from 2024 and 2025 and the first half of 2026 says something consistently across dozens of independent research groups, across FPGA implementations and ASIC measurements and theory papers, that consistency is itself a signal. The field is converging. The question is where.

What follows is built on three categories of evidence. First, published silicon measurements — not simulations, not theoretical projections, but actual synthesized hardware with real area, power, and throughput numbers. Second, the structure of the problem — the mathematical and architectural reasons why certain approaches fit the constraints of modern AI inference better than others. Third, the trajectory of the research community itself, which has gone from a handful of CORDIC-for-neural-networks papers per year in 2020 to a continuous stream of independent validation across multiple continents, conferences, and research traditions in 2025–2026.

When hardware measurements, structural argument, and research momentum all point in the same direction, the forecasting burden shifts. You are no longer trying to predict what will happen. You are trying to explain why people haven't noticed what is already true.

---

## The Baseline: What Is Already True

Before predicting what comes next, the ground state must be fixed. As of June 2026, here is what has been established, not hypothetically but in silicon.

CORDIC-based activation function cores are outperforming dedicated implementations across the full AI function portfolio — sigmoid, tanh, GeLU, Swish, SELU, SoftMax, ReLU — while being reconfigurable between all of them at runtime. The DA-VINCI core demonstrated 4.5× LUT reduction and 5.4× power reduction on FPGAs, and 16.2× area reduction on 45nm CMOS, with 98.5% accuracy parity vs. floating-point. These numbers have been independently reproduced in the IIT Indore research cluster's subsequent architectures.

CORVET, published February 2026, achieved 4.83 TOPS/mm² compute density and 11.67 TOPS/W energy efficiency in a 256-PE CORDIC-based configuration, with 4× throughput improvement and 33% cycle reduction per MAC stage versus prior art. This is not a research prototype optimized for a single benchmark — it is a general-purpose vector processing engine validated on both CNN and transformer-style MLP workloads.

CARMEN, also published in 2026, delivers the same CORDIC-accelerated multi-precision inference at 154.6ms latency for object detection at 0.43W on a Pynq-Z2 FPGA — real-time edge inference on a device that costs twenty dollars and consumes less than half a watt.

The SYCore systolic CORDIC engine has demonstrated 4.64× throughput improvement at 5.02× power reduction at 28nm CMOS across transformers, RNNs/LSTMs, and DNNs simultaneously — one chip, one arithmetic primitive, all workloads.

The Logarithmic Posit-enabled Reconfigurable edge-AI Engine (LPRE) from ISCAS 2025 has demonstrated that CORDIC and posit number systems are not competing alternatives but complementary layers of the same hardware abstraction, with the CORDIC fixed-point activation function block serving as the numerical backend for posit-encoded weights.

All of this — every result cited above — was published between February 2025 and May 2026. The field is not converging slowly. It is converging fast.

That is the baseline. Now, twelve predictions for where it goes.

---

## Prediction 1: Dedicated RoPE Hardware Built on CORDIC Circular Mode Ships in a Production AI ASIC Before the End of 2027

**Confidence: 87%**

**The claim.** At least one major custom AI ASIC — from a hyperscaler, a startup, or a semiconductor company — will ship a production chip before December 2027 with a dedicated hardware block for Rotary Position Embedding computation, built on CORDIC circular rotation mode, as a first-class architectural feature rather than a software workaround.

**The evidence chain.** RoPE is now the positional encoding of record for essentially every significant open-weight language model: Llama 3, Qwen 2/3, Mistral, Gemma, DeepSeek, Falcon. It is applied at every transformer layer, for every token, across all attention heads — in a model with 80 layers and 64 heads, generating one token requires over five thousand 2D rotation operations. This is not a marginal operation. It is structurally central to the forward pass of every modern LLM.

On current GPU hardware, RoPE is computed using precomputed sin/cos tables that are broadcast across the batch. This is a memory bandwidth operation disguised as a compute operation: the tables live in DRAM or L2 cache, they are fetched per-token, and the cost scales linearly with context length. For inference at 128K or 256K token contexts — which is already standard in production deployments — this is a non-trivial fraction of total memory traffic.

CORDIC circular rotation mode is mathematically equivalent to RoPE computation: it takes an angle and a 2D input vector and produces the rotated output using shifts and additions. There is no approximation; the rotation is exact to the precision of the iteration count. For INT8 inference — the dominant mode in 2026 custom silicon — 10 CORDIC iterations produce accuracy well within the quantization noise floor. The hardware cost is a handful of adders. No lookup tables. No DRAM traffic for sin/cos values.

The economics are clear. A dedicated RoPE-CORDIC block added to a custom AI ASIC costs a fraction of one percent of the total die area. It eliminates sin/cos table fetches, which at long context lengths represent a measurable fraction of memory bandwidth. For a hyperscaler running inference at millions of queries per day, the return on that die area is immediate.

The research literature first identified the CORDIC–RoPE structural equivalence in 2024–2025, and noted explicitly that no production silicon implementation had been reported. That observation, in a fast-moving field with a three-year chip cycle, is a flag planted in 2024 pointing to a product arriving in 2027.

**The failure condition.** This prediction fails if LLM architectures shift away from RoPE at scale — for instance, if ALiBi-style position encodings or NoPE (no positional encoding) architectures gain wide adoption, or if a fundamentally different position encoding with no rotation structure becomes dominant. The evidence against this failure is that RoPE's dominance deepened, not weakened, through 2025 and 2026, and that multimodal extensions (M-RoPE for video, spatial 2D/3D RoPE for vision-language models) are expanding the RoPE computational footprint, not contracting it.

---

## Prediction 2: CORDIC-Based Softmax Displaces LUT-Based Exponential Approximation in All New Quantized Custom AI Chip Designs by 2028

**Confidence: 79%**

**The claim.** By the end of 2028, the dominant implementation strategy for softmax exponential computation in newly designed quantized AI inference chips will be CORDIC-based (specifically HGH-CORDIC or its successors), not lookup-table-based polynomial approximation. The transition will be traceable in conference proceedings and patent filings.

**The evidence chain.** The softmax function drives INT8 and INT4 inference into one of the most awkward corners of digital hardware. It requires the exponential function `e^x`. Dedicated floating-point exponential units are expensive and power-hungry. Lookup tables for `e^x` require BRAM resources that compete with weight buffers. Polynomial approximations are fast but fail at the tails of the input distribution, causing occasional catastrophic accuracy loss that is difficult to detect and debug at inference time.

CORDIC hyperbolic mode computes `e^x` exactly (to the precision of the iteration count) using shifts and additions. HGH-CORDIC, the high-radix generalized hyperbolic variant published at IEEE ARITH 2024, reduces the iteration count by more than 50% versus radix-2 variants for the same accuracy target, and achieves 40% area and power savings at 28nm. The VEXP work (arXiv 2504.11227, April 2025) demonstrated HGH-CORDIC as the backend for a RISC-V ISA extension that accelerates softmax by 6–8× with under 3% total core area overhead.

The timing matters. New custom AI chip designs that tape out in 2026 and 2027 are being architected right now, and the ARITH 2024 result, the VEXP 2025 result, and the CORVET/CARMEN 2026 results collectively close the performance gap between CORDIC and LUT-based approaches — while CORDIC wins comprehensively on reconfigurability, because the same hardware that computes the softmax exponential also computes sigmoid, GeLU, and RoPE rotations.

The mixed-precision reality of 2026 inference — where different layers use FP8, INT8, or INT4, and where the format choice may shift between inference passes for the same model — makes reconfigurability not a luxury but a necessity. A LUT table sized for FP8 does not reuse for INT8. A CORDIC unit sized for 8-bit precision serves both with mode switching.

**The failure condition.** This prediction fails if a radically simpler softmax approximation — one that is both accurate enough and cheap enough to out-compete CORDIC on all dimensions — emerges in the 2026–2027 timeframe. The leading candidate would be a linear softmax approximation (replacing exp with 2^x bitshifts, sometimes called "base-2 softmax"). Base-2 softmax does exist and is already used in some hardware, but it introduces distributional shifts that compound across transformer layers in ways that degrade long-sequence coherence. The community is aware of this failure mode, which reduces the probability that base-2 softmax displaces CORDIC in anything but the most latency-critical, accuracy-tolerant deployments.

---

## Prediction 3: CORDIC Arithmetic Appears in the LLM Training Stack, Not Just Inference, by 2029

**Confidence: 61%**

**The claim.** By 2029, at least one peer-reviewed work from a Tier 1 ML research institution or major hardware vendor will demonstrate CORDIC-based computation in the forward or backward pass of LLM training — not inference — with quantitative evidence that the accuracy cost is acceptable for pre-training at scale.

**The evidence chain.** CORDIC in the training stack is the most speculative of the near-term predictions, and the lower confidence level reflects genuine uncertainty. The argument for it, however, is structural rather than speculative.

Training requires higher numerical precision than inference. The standard is BF16 or FP16 for activations, with FP32 accumulation in the optimizer. CORDIC has been characterized at BF16 precision (ISQED 2024), and 16-bit CORDIC for hyperbolic functions — 16–18 iterations — delivers accuracy within the BF16 format's own rounding error. The training bottleneck for nonlinear functions is not precision loss; it is throughput, and CORDIC's pipelined implementation is throughput-competitive with dedicated transcendental units at 16-bit precision.

The more compelling path, however, is not replacing FP16 exponentials with CORDIC exponentials — it is using CORDIC for the backward-pass activation function gradients. The gradients of sigmoid, tanh, and GeLU are smooth functions of the forward-pass output: `σ'(x) = σ(x)(1 - σ(x))`, `tanh'(x) = 1 - tanh²(x)`. A hardware unit that computes the forward sigmoid or tanh (via CORDIC) can compute the gradient with one additional multiply (which CORDIC linear mode handles) and one subtraction. The same CORDIC datapath computes both forward and backward pass nonlinear operations, with no additional hardware.

The FP8 training trajectory matters here too. The trajectory from FP32 training to FP16 training to BF16 training to FP8 training has been consistent and directional. As FP8 training becomes standard — and the 2025–2026 evidence from Nvidia Hopper and successor platforms suggests it will, with appropriate scaling factor management — the precision requirements for activation function computation drop to exactly the range where CORDIC is native.

**The failure condition.** This prediction fails if the training stack converges on hardware exponential units (dedicated silicon for `e^x` at BF16 precision) as a standard IP block across the industry before the CORDIC alternative reaches the same performance target. This is possible if the major GPU vendors — Nvidia, AMD — standardize FP8 exponential units in ways that prevent displacement. The probability is roughly 40%, which is why the confidence is 61%.

---

## Prediction 4: A CORDIC-Aware Compiler Backend Ships as a First-Class Target in an Open-Source ML Compilation Framework by the End of 2028

**Confidence: 74%**

**The claim.** By December 2028, TVM, MLIR, or their successors will include a first-class CORDIC hardware backend — not a user-supplied custom operator, but a standard target — that automatically maps `tf.exp`, `tf.sigmoid`, `tf.tanh`, `jax.nn.gelu`, and similar operations to CORDIC hardware when targeting RISC-V or FPGA backends.

**The evidence chain.** The gap between what the hardware can do and what the software stack knows about it is currently large. CORDIC-based AI chips — SYCore, CORVET, CARMEN, DA-VINCI-enabled edge devices — all require manual integration with training and deployment frameworks. The QKeras/FxPmath emulation layer used in DA-VINCI and CORVET validation is a research scaffold, not a production compiler.

This gap will close because the economics demand it. A chip vendor building a CORDIC-accelerated edge NPU cannot ship it without a compiler that automatically utilizes the CORDIC blocks. Without compiler support, users must hand-write assembly-level CORDIC invocations, which is a nonstarter for any deployment at scale. Compiler support is not a nice-to-have for CORDIC-based silicon; it is the precondition for market adoption.

The MLIR ecosystem, which now underlies TPU compilation (Google's OpenXLA), Triton (used for Nvidia GPU kernels), and several custom AI chip backends, is structurally well-suited to CORDIC integration. MLIR's lowering passes can represent a `math.exp` operation at the abstract level and progressively lower it through intermediate representations to CORDIC hardware instructions. The infrastructure exists. The missing piece is the CORDIC lowering pass itself.

The VEXP work (April 2025) demonstrated that a RISC-V ISA extension for softmax can be targeted by an LLVM toolchain with 6–8× latency improvement and minimal change to the compilation pipeline. This is exactly the template for a compiler backend: the hardware extension is defined, the ISA encoding is fixed, and the compiler maps high-level operations to it. The remaining work is engineering, not research.

The timeline to 2028 reflects the realistic pace of compiler ecosystem evolution: a proof-of-concept targeting one framework appears in 2026 or 2027, it is upstreamed to the main project after community validation, and the first stable release ships in 2028.

**The failure condition.** This prediction fails if the CORDIC-accelerated chip market fragments into too many incompatible ISA extensions for a single compiler target to be viable. If every vendor defines their CORDIC integration differently — different precision formats, different mode encodings, different pipeline depths — then the compiler backend reduces to a vendor-specific plugin rather than a universal target. This fragmentation is exactly what happened with early FPGA AI toolchains, and it is not a negligible risk. The mitigation is the RISC-V standardization trajectory described in Prediction 5.

---

## Prediction 5: An Open RISC-V CORDIC Extension Standard Is Defined and Receives Two or More Independent Implementations Before 2029

**Confidence: 68%**

**The claim.** By 2029, a formal RISC-V ISA extension specification for CORDIC-based computation — analogous to the existing RISC-V B (bit manipulation) or V (vector) extensions — will be proposed through the RISC-V International working group process and will have at least two independent open-source hardware implementations.

**The evidence chain.** RISC-V is the dominant ISA for custom AI inference silicon at the edge. It is the control processor in SYCore, the host for the VEXP softmax extension, the basis for the CORDIC-DMA coprocessor (MDPI 2022), and the core in dozens of edge AI systems from academic and commercial research groups worldwide. Every one of these implementations has independently developed its own CORDIC hardware interface, its own instruction encoding, and its own software toolchain glue.

This situation — many independent implementations of the same hardware concept with incompatible interfaces — is exactly the situation that motivates RISC-V extension standardization. The RISC-V extension process is well-understood: a task group proposes a specification, multiple parties implement it independently (both in hardware and in software), the implementations are validated against the specification, and the extension is ratified. The B extension took approximately three years from initial proposal to ratification. The V extension took four years.

The CORDIC case is structurally simpler than either B or V because the hardware is iterative (fewer instructions to specify), the precision model is well-established (fixed iteration count determines precision), and the use cases are narrow and well-defined (transcendental functions for AI inference). A minimal CORDIC extension might specify as few as four instructions: one for each combination of {circular, hyperbolic} × {rotation, vectoring} mode, with precision controlled by a configuration register.

The IIT Indore research cluster — which produced RECON, DA-VINCI, SYCore, CORVET, CARMEN, LPRE, EULER-ADAS, and QForce-RL — is the most natural anchor for such a standardization effort, given its depth of published implementation experience. The group's active collaboration with European partners (including Adam Teman at Bar-Ilan University) and its consistent publication track record suggest both the motivation and the institutional credibility to lead a standardization proposal.

**The failure condition.** This prediction fails if the RISC-V extension standardization process loses momentum relative to the speed of commercial silicon development. If major chip vendors standardize on proprietary CORDIC interfaces before the RISC-V process concludes — locking in incompatible ISAs that cannot be harmonized — the open standard becomes irrelevant. This is the plausible failure scenario, with roughly 32% probability.

---

## Prediction 6: CORDIC-Based Processing Elements Achieve 10 TOPS/W on Edge AI Workloads in Production Silicon at 5nm or Below by 2028

**Confidence: 82%**

**The claim.** By the end of 2028, a production CORDIC-based edge AI processor will demonstrate 10 TOPS/W or better on standard edge AI benchmarks (MLPerf Tiny or equivalent), synthesized in 5nm or finer process technology.

**The evidence chain.** CORVET at 28nm achieved 11.67 TOPS/W in a 256-PE configuration (February 2026). CARMEN at 28nm achieved 4.83 TOPS/mm² compute density. These are 28nm numbers.

The standard projection from 28nm to 5nm process technology — accounting for transistor scaling, supply voltage reduction, and improved cell libraries — implies roughly 5–8× improvement in energy efficiency for the same digital logic. A CORDIC array that achieves 11.67 TOPS/W at 28nm should, by Moore's law projections, achieve 58–93 TOPS/W at 5nm.

That projection is optimistic for several reasons — interconnect scaling, memory access dominance at smaller nodes, and the fact that the CORVET measurements reflect the full system including memory overhead. A conservative estimate — discounting 70% of the Moore's law projection — still places a 5nm CORDIC-based edge processor above 15 TOPS/W.

The 10 TOPS/W threshold is the relevant comparison point because it corresponds to the energy efficiency range where battery-powered wearable and medical devices become practical for continuous on-device inference. At 10 TOPS/W, a 100mW power budget (generous for a wearable) supports 1 TOPS of inference compute — enough for continuous voice activity detection, basic vision models, or EEG artifact detection running at 10–100 Hz.

The 5nm timeline is grounded in the industry's process node roadmap. TSMC N5 and Samsung 5LPE are mature production nodes as of 2026. Edge AI chips fabbed at N5 are not speculative — Apple's A16 chips and Qualcomm's Snapdragon 8 Gen 2 are both N4 or N5 class. The question is whether a purpose-built CORDIC-centric edge AI chip reaches N5 production by 2028. Given the current research velocity and typical 24–30-month tape-out cycle, a design that starts now for N5 production would tape out in late 2026 or early 2027 and ship in 2028.

**The failure condition.** This prediction fails if the CORDIC-based approach faces insurmountable challenges in the N5 design rule environment — for instance, if the shift-and-add logic is less area-efficient than expected relative to N5-optimized multiply-accumulate hard macros, or if memory access overhead dominates to the point where the compute efficiency of CORDIC is irrelevant to system-level energy. The latter is the more plausible failure mode, and it motivates Prediction 7.

---

## Prediction 7: The First "CORDIC-Native" Neural Architecture — Designed Specifically for CORDIC Arithmetic — Achieves Publication in a Top-Tier ML Venue by 2028

**Confidence: 58%**

**The claim.** By 2028, a neural network architecture designed from the ground up with CORDIC hardware constraints in mind — choosing activation functions, normalization approaches, and attention variants that map natively to CORDIC shift-and-add arithmetic — will be published at NeurIPS, ICML, ICLR, or CVPR, and will demonstrate competitive accuracy with FP32 baselines on at least one standard benchmark.

**The evidence chain.** The history of hardware-software co-design in AI has a consistent pattern: first, the hardware is adapted to the existing algorithms (GPU tensor cores for transformer attention). Then, the algorithms begin adapting to the hardware (FlashAttention for IO-aware attention, INT8 quantization for tensor core efficiency). The first phase is already underway for CORDIC. The second phase — algorithmic adaptation — is what this prediction anticipates.

The specific adaptations are not difficult to envision. An architecture that replaces GeLU with tanh-based gating — not because tanh is theoretically superior, but because tanh is CORDIC's native function — would sacrifice essentially nothing in model quality (the empirical gap between GeLU and tanh activations is small) while gaining exact, hardware-efficient computation. An architecture that replaces standard LayerNorm with RMSNorm backed by CORDIC reciprocal square root, with fixed-point accumulation, maps cleanly onto CORDIC linear and vectoring modes. An attention mechanism that uses base-2 softmax — computable by CORDIC hyperbolic mode with reduced precision cost — with appropriate temperature compensation represents a similarly minor algorithmic change with significant hardware benefit.

The CORDIC-native architecture family does not need to be dramatically different from existing architectures. It needs to be different enough to demonstrate the hardware efficiency gains, similar enough to existing baselines to compete on accuracy, and novel enough to constitute a publishable contribution. This is a low bar relative to the state of the field.

The reason the confidence is 58% rather than higher is publication venue risk: a CORDIC-native architecture paper requires either a strong ML venue (which tends to prioritize accuracy improvements over hardware efficiency) or a strong hardware venue (which may not recognize the architectural novelty). The sweet spot — a paper that frames CORDIC-native design as a general principle for hardware-efficient AI, demonstrated at scale — requires authors with both ML and hardware credibility, which is a rare combination. It exists in the IIT Indore cluster and in groups like the EDLAB at EPFL, but the ML-focused framing may not come naturally from hardware-first researchers.

**The failure condition.** This prediction fails if the ML community treats CORDIC-native design as a hardware implementation detail rather than an architectural choice — effectively delegating it entirely to the hardware layer and never engaging with it at the model architecture level. This would mean CORDIC-native networks are built but never published in ML venues, appearing only in circuits and systems journals where the framing is hardware efficiency rather than model design. This is a plausible failure mode, making the 58% confidence appropriate.

---

## Prediction 8: Processing-in-Memory AI Chips Standardize on CORDIC as the Universal Post-Conversion Nonlinear Block by 2028

**Confidence: 76%**

**The claim.** By 2028, the dominant architecture for the analog-to-digital conversion boundary in compute-in-memory (CIM) AI accelerators will include a CORDIC-based digital nonlinear processing block as a standard component, appearing in both academic publications and commercial IP cores.

**The evidence chain.** The compute-in-memory paradigm — performing matrix-vector multiplication inside SRAM or RRAM arrays, eliminating the DRAM bandwidth bottleneck — faces a fundamental architectural problem: the analog computation produces an analog or low-resolution digital result that must be followed by nonlinear activation before the next layer. The analog-to-digital conversion at this boundary is the primary energy cost of CIM architectures. Once the result is in digital form, some nonlinear function must be applied before re-encoding for the next computation.

The 2024 CIM literature quantifies this precisely: CORDIC-based digital activation blocks placed immediately after the ADC reduce AD/DA energy by up to 12.31× and improve overall energy efficiency by 2.34–5.20× relative to alternative approaches. The reason is structural: CORDIC uses the same digital fabric as the rest of the CIM chip, does not require reference voltages calibrated to specific functions, and scales with process technology in the same way as the digital control logic. It is technology-agnostic in a way that analog activation function circuits are not.

As CIM moves from academic demonstrations to production silicon — and the 2025–2026 activity from d-Matrix, Untether AI, Hailo, and academic groups suggests this transition is underway — the architectural decisions crystallize. The question of "what happens to the signal after the ADC" must be answered in the tape-out decision. CORDIC provides an answer that is simultaneously area-efficient, power-efficient, reconfigurable across all activation functions, and directly integrable with digital control logic. No analog approach has all four properties simultaneously.

The prediction is about standardization: the claim is not that CORDIC will be used in CIM designs (it already is, in research silicon) but that it will become the expected default — the approach that a CIM chip designer would choose without requiring justification, just as a conventional DNN accelerator designer chooses a systolic array for GEMM.

**The failure condition.** This prediction fails if analog activation function circuits mature faster than expected — if analog sigmoid and tanh implementations achieve the accuracy and robustness needed for production deployments, eliminating the need for a digital activation stage entirely. This would represent a genuine paradigm shift (true end-to-end analog computation), which has been repeatedly announced and repeatedly deferred. The failure probability for this path is roughly 24%.

---

## Prediction 9: CORDIC and Posit Arithmetic Merge Into Unified Number Format Processing Engines That Displace Traditional Fixed-Point-Only Designs in Edge AI by 2029

**Confidence: 65%**

**The claim.** By 2029, the dominant edge AI processing engine architecture will be a unified CORDIC/posit hybrid — using posit number format for weight representation and accumulation, and CORDIC for activation function computation — rather than the traditional fixed-point/LUT combination. This prediction is specifically about edge AI (sub-5W) deployments, not data center inference.

**The evidence chain.** The LPRE (Logarithmic Posit-enabled Reconfigurable edge-AI Engine, ISCAS 2025) is the paper that makes this prediction possible rather than speculative. It demonstrated for the first time that posit arithmetic for weight representation and CORDIC for activation functions are not alternatives but complements: the posit-encoded weights feed into a fixed-point conversion stage that interfaces natively with the CORDIC activation function block. The accuracy results match INT8 and FP16 baselines on standard vision workloads.

Why posit? The posit number format — proposed by John Gustafson in 2017 — provides higher accuracy than IEEE 754 floating-point in the range of numbers that neural network weights actually occupy (near zero and mid-range values), while using fewer bits. For edge AI where memory bandwidth and storage are primary constraints, posit's variable-precision distribution over the number line is a structural advantage that fixed-point cannot replicate.

Why CORDIC with posit? Posit arithmetic shares CORDIC's philosophical core: both systems trade uniform precision for adaptive precision, both are natively iterative in their hardware implementation, and both are designed for environments where traditional floating-point hardware is too expensive. The RAMAN architecture (VLSID 2026) — "Resource-efficient Approximate Posit Processing for Algorithm-Hardware Co-design" — extends the LPRE insight further, demonstrating posit-CORDIC co-design across a broader range of edge AI workloads.

The EULER-ADAS architecture (May 2026) demonstrates the same unified posit/CORDIC approach in an automotive context — ADAS (Advanced Driver Assistance Systems) — where the functional safety requirements mandate high accuracy but the power envelope is severely constrained. This is the most demanding possible validation environment for a new arithmetic paradigm.

The cumulative evidence from LPRE, RAMAN, EULER-ADAS, and CORVET is that the posit-CORDIC combination is not a research curiosity but a coherent engineering solution that multiple independent groups have reached from different starting points. Convergence from independent sources is the most reliable indicator available.

**The failure condition.** This prediction fails if the posit number format fails to achieve industry adoption — if chip vendors, compiler teams, and framework developers do not support posit in ways that make it accessible to system designers. This is a real risk. Posit has been proposed before and has not yet crossed into mainstream silicon. The CORDIC-posit prediction is only as strong as the posit adoption prediction, which is the most uncertain link in the chain. Confidence at 65% rather than 75% reflects this coupling.

---

## Prediction 10: The IIT Indore NSDCS Lab Cluster Becomes a Recognized IEEE Standards-Adjacent Reference Point for CORDIC AI Hardware by 2027

**Confidence: 88%**

**The claim.** By 2027, the NSDCS Lab at IIT Indore (led by Prof. Santosh Kumar Vishvakarma), together with its collaborators, will be recognized in IEEE TVLSI, IEEE TCAS, and/or ACM TRETS editorial contexts as the primary reference research group for CORDIC-based AI acceleration — cited in the introduction of papers from groups with no direct collaboration as the canonical prior work. This recognition will be visible in citation patterns and editorial board participation.

**The evidence chain.** This is not a prediction about a technical outcome. It is a prediction about the sociological dynamics of a research field, which are often more predictable than the technical dynamics.

The NSDCS Lab has produced RECON (2021), DA-VINCI (2025), SYCore (2025), DA-VINCI retrospective (ISVLSI 2025), CORVET (2026), CARMEN (2026), LPRE (ISCAS 2025), QForce-RL (VDAT 2025), EULER-ADAS (2026), TREA (2026), and POLARON (2025) — more than ten significant CORDIC-for-AI papers across two years, across four leading conferences and two leading journals, with a coherent intellectual lineage that builds each result on the previous one. This publication velocity is extraordinary.

It is accompanied by international collaboration (Adam Teman at Bar-Ilan University is a recurring co-author), which increases the cross-institutional credibility of the results, and by an explicit patent application (US Patent App. 18/534,035, April 2024) that signals commercial seriousness. The combination of academic rigor, international collaboration, commercial intent, and publication volume matches precisely the profile of research groups that become field-defining centers.

The sociological prediction is simply that the field recognizes what is already true: this cluster has done more to establish CORDIC as a serious AI hardware primitive than any other group in the world, and the citation patterns of 2027 will reflect that.

**The failure condition.** This prediction fails if a larger, more well-resourced group — from MIT, Stanford, CMU, ETH Zurich, or a major semiconductor company's research lab — enters the CORDIC-for-AI space in 2026–2027 with results that overshadow the IIT Indore work. This is possible. The field is attracting attention, and large research groups are responsive to attention. At the moment, no such group has engaged with CORDIC-for-AI at the depth the IIT Indore cluster has. If that changes, the 88% confidence revises downward to approximately 60%.

---

## Prediction 11: CORDIC-Based Approximate Computing Frameworks Demonstrate Full Training and Inference Pipelines for 1B+ Parameter Models with Less Than 2% Accuracy Degradation by 2029

**Confidence: 55%**

**The claim.** By 2029, a published full pipeline — training and inference — for a model with 1 billion or more parameters will demonstrate CORDIC-based nonlinear computation throughout, with end-to-end accuracy degradation below 2% on standard benchmarks relative to FP32 baselines.

**The evidence chain.** This prediction is more ambitious than the others, and the lower confidence reflects genuine technical uncertainty rather than sociological uncertainty. The argument for it is that every individual component has been demonstrated at smaller scale: CORDIC-based activation functions at 98.5% QoR, CORDIC-based softmax, CORDIC-based normalization, and CORDIC-based position embeddings have all been validated independently. The remaining step is integration at billion-parameter scale.

The integration challenge is not primarily hardware — it is numerical stability. CORDIC arithmetic introduces structured approximation errors at each nonlinear operation. In a 30-layer network, those errors compound. The question is whether the compounding is bounded in a useful way.

The evidence from quantization research suggests it is. INT8 inference — which introduces approximation errors at least as large as CORDIC's — routinely achieves less than 1% accuracy degradation at 1B+ parameter scale with appropriate calibration. CORDIC's error profile is, if anything, more structured and predictable than quantization noise: it is dominated by the scale factor error (fixed and correctable) and the convergence tail (bounded by the iteration count). A CORDIC-aware calibration procedure — analogous to post-training quantization calibration — should be able to compensate for these errors at the model level.

The 2029 timeline is driven by the hypothesis that CORDIC-aware training pipelines will be developed in 2026–2027, demonstrated at 100M-parameter scale in 2027–2028, and scaled to 1B parameters in 2028–2029. This is a plausible but not certain trajectory. The 45% failure probability reflects the genuine uncertainty about whether numerical stability at 1B-parameter scale is achievable without the kind of fundamental architecture modification described in Prediction 7.

**The failure condition.** This prediction fails if CORDIC approximation errors compound in a qualitatively different way at billion-parameter scale than at hundred-million-parameter scale — specifically, if there are activation patterns in large models that systematically drive CORDIC into convergence tail behavior across multiple layers simultaneously. This is the "adversarial input for CORDIC" failure mode, and it is not hypothetical: certain extreme activation patterns (very large positive values in attention logits, for instance) can exceed CORDIC's convergence range and require pre-conditioning, which adds latency and reduces the hardware efficiency advantage. At small model scales, these events are rare. At billion-parameter scales, with more complex input distributions, they may be less rare.

---

## Prediction 12: Power-Constrained AI Markets — Hearing Aids, Implantables, Industrial Sensors — Converge on CORDIC as the De Facto Standard for On-Device Inference by 2028

**Confidence: 84%**

**The claim.** By 2028, CORDIC-based nonlinear computation will be the dominant approach in new chip designs targeting sub-10mW on-device AI inference — hearing aids, implantable neural interfaces, industrial condition monitoring, and environmental sensing. This convergence will be visible in ISSCC/CICC/ISCAS publications and in commercial product announcements.

**The evidence chain.** This prediction carries the highest confidence outside the citation prediction, for two reasons that reinforce each other: the physics of ultra-low-power AI is extremely constraining, and CORDIC is uniquely matched to those constraints.

Below 10mW total system power, the arithmetic choices available to a chip designer collapse. Floating-point units are eliminated — even a 16-bit FPU at advanced nodes costs 5–10mW in dynamic power alone. Lookup tables require SRAM that is both expensive in area and in access energy. Polynomial approximation requires multipliers, which at these power levels are allocated entirely to the MAC operations and cannot be shared with activation function computation.

The shift-and-add CORDIC operation costs roughly 0.1–0.5 pJ per bit-operation at N7 or below, depending on the cell library and clock frequency. For a 10-stage CORDIC computing tanh at 8-bit precision, the total energy per activation evaluation is approximately 1–5 pJ. This is within the energy budget of a 100Hz-continuous inference engine operating at 1mW total.

No alternative approach achieves this. LUT-based approaches require SRAM reads (approximately 1–2 fJ/bit read, but 256 bits per 8-bit address, so ~0.5 pJ per lookup for 8-bit precision — competitive, but lacking reconfigurability and requiring SRAM area that competes with weight storage). Polynomial approximation with dedicated multipliers requires 3–10 pJ per multiply at N7, which does not scale below 5mW system power when multiplied by the number of activation evaluations per inference step.

The hearing aid and implantable markets have additional constraints that make CORDIC attractive: reliability (CORDIC has no failure modes from table corruption or coefficient rounding), testability (the deterministic shift-and-add structure is straightforward to verify in silicon), and radiation tolerance (shift and add logic is among the most robust digital logic to single-event upsets, relevant for satellite and some medical deployments).

The STM32 CORDIC coprocessor — already in mass production in the STM32G4 and STM32H7 families — is the leading edge of this market, bringing CORDIC into the embedded system design toolbox for millions of engineers worldwide. The next generation of purpose-built sub-mW AI chips will extend what the STM32 coprocessor demonstrated at 170MHz to purpose-built inference engines at frequencies and power levels an order of magnitude lower.

**The failure condition.** This prediction fails if ultra-low-power AI inference converges on spike-based or spiking neural network (SNN) computation rather than conventional DNN inference. SNNs eliminate most nonlinear activation function computation (spike events are binary), which would eliminate the market need for CORDIC. The probability of this failure mode is approximately 16%, reflecting genuine uncertainty about whether SNN-based edge inference achieves competitive accuracy on the workloads (keyword spotting, anomaly detection, vital sign monitoring) that dominate the sub-10mW market.

---

## The Counter-Predictions: What Won't Happen

Intellectual honesty demands a section on failure modes at the systemic level — not just the failure conditions for individual predictions, but the broader scenarios under which the entire framework breaks down.

CORDIC does not win in every scenario. It wins specifically under the conditions that define AI inference today: quantized precision, reconfigurable function requirements, power-constrained deployments, and systolic array architectures. If those conditions change fundamentally, the case for CORDIC changes with them.

The most credible systemic challenge is photonic computing. Optical matrix-vector multiplication is orders of magnitude more energy-efficient than digital GEMM at large scale, and it changes the arithmetic bottleneck entirely. In a photonic computing system, the nonlinear activation function is the dominant compute cost (because it requires electro-optic conversion), and the character of that cost is fundamentally different from digital CORDIC. Lightmatter and other photonic compute startups are serious engineering organizations with real silicon. If photonic computing achieves the accuracy and reliability needed for general AI inference within the decade — a significant if, but not an impossible one — then the future of AI nonlinear computation is optical modulators and photodetectors, not shift-and-add logic.

A second systemic challenge is the possibility of neural network architectures that eliminate transcendental functions entirely. Linear attention approximations (replacing softmax attention with linear kernel functions), ReLU-based attention (Wortsman et al., 2023), and NoPE architectures (no positional encoding, eliminating RoPE) are all active research directions. If any of these converges into a model family that matches transformer performance without requiring sigmoid, tanh, exp, or sin/cos, then CORDIC's primary value proposition — computing those functions cheaply — becomes irrelevant. The evidence from 2025–2026 does not support this trajectory: softmax attention remains dominant, RoPE is deepening rather than retreating, and GeLU has replaced simpler activations. But the research space is open.

The third systemic challenge is the development of cheap, reliable lookup-table-free floating-point exponential units. This is the "same problem, opposite solution" failure mode. If NVIDIA, AMD, or a custom ASIC vendor develops a dedicated `e^x` hardware unit at 4-bit precision that is as area-efficient as CORDIC's hyperbolic mode but faster and simpler, CORDIC loses the softmax battle on purely technical grounds. This is unlikely at FP4 or INT4 precision (the error structure of floating-point exponential approximation at 4 bits is very poor), but not impossible at FP8 with careful scaling.

---

## The Meta-Thesis: Why CORDIC Keeps Coming Back

These twelve predictions are grounded in individual technical and market trajectories, but they share a common structure that is worth making explicit.

CORDIC keeps coming back — from the B-58 bomber to the HP scientific calculator to the FPGA signal processing era to the AI accelerator era — because it is matched to a class of problems, not a specific problem. The class is: compute transcendental functions in constrained hardware that cannot afford dedicated floating-point units.

The hardware of constrained AI inference is, in every meaningful dimension, exactly this class. The constraints that define AI inference silicon in 2026 — power budgets in milliwatts, area budgets in square millimeters, precision requirements in 8 bits or below, reconfigurability requirements across dozens of function types — are the same constraints that make CORDIC optimal. Not better than the alternatives by a small margin. Structurally better, because the alternatives require resources (multipliers, BRAMs, floating-point units) that are simply unavailable.

The meta-prediction, then, is not about a specific architecture or a specific benchmark. It is about the structure of the problem. As long as AI inference silicon operates under these constraints — as long as power is scarce, area is expensive, precision is limited, and the function set is evolving — CORDIC will be the right answer. And those constraints are not going away. They are the physics of the devices, the economics of the processes, and the market requirements of the applications. They are as fundamental as the bomber's navigation problem was in 1957.

Sixty-seven years later, Jack Volder's insight still holds. The only thing that has changed is the scale of the problem it solves.

---

## The Calibration Table

| Prediction | Confidence | Timeline | Primary Failure Mode |
|---|---|---|---|
| 1. Dedicated RoPE-CORDIC hardware in production ASIC | 87% | By Dec 2027 | RoPE architectural displacement |
| 2. CORDIC softmax displaces LUT-exp in quantized designs | 79% | By Dec 2028 | Base-2 softmax adoption |
| 3. CORDIC enters LLM training stack | 61% | By Dec 2029 | FPU exponential standardization |
| 4. CORDIC compiler backend in open-source ML framework | 74% | By Dec 2028 | ISA fragmentation |
| 5. Open RISC-V CORDIC extension ratified | 68% | By Dec 2029 | Commercial proprietary lock-in |
| 6. 10 TOPS/W CORDIC edge silicon at 5nm | 82% | By Dec 2028 | Memory access dominance |
| 7. CORDIC-native architecture in top ML venue | 58% | By Dec 2028 | ML/hardware community separation |
| 8. CIM chips standardize on CORDIC post-ADC | 76% | By Dec 2028 | Analog activation maturation |
| 9. CORDIC-posit unified edge AI engine | 65% | By Dec 2029 | Posit ecosystem failure |
| 10. IIT Indore as recognized field reference center | 88% | By Dec 2027 | Large-group competitive entry |
| 11. 1B-param CORDIC pipeline under 2% degradation | 55% | By Dec 2029 | Adversarial activation compounding |
| 12. Sub-10mW market convergence on CORDIC | 84% | By Dec 2028 | SNN edge AI adoption |

---

## What This Means for Practitioners

For someone designing an edge AI chip today, these predictions carry a specific practical implication: the decision to include or exclude a CORDIC block is not a preference. It is a structural choice about whether the chip is aligned with where the field is going.

A chip designed without CORDIC capability will face a growing gap with the research frontier. It will be less reconfigurable as the activation function landscape continues to evolve. It will handle RoPE computation less efficiently as long-context inference becomes standard. It will be harder to integrate with the compiler backends that are being developed for CORDIC-capable silicon. And it will miss the energy efficiency gains at the MAC level that CORVET, CARMEN, and SYCore have demonstrated are available.

For a researcher working on AI hardware, the predictions suggest a specific set of problems that are underworked: CORDIC-aware quantization-aware training, CORDIC numerical stability analysis at scale, CORDIC-MLIR compiler lowering passes, and CORDIC-native architecture search. These are not adjacent to the field's current direction — they are the field's current direction, stated as engineering problems rather than research problems.

For a system architect evaluating AI silicon for deployment, the predictions suggest that the energy efficiency trajectory of CORDIC-based designs, grounded in the CORVET and CARMEN measurements, should inform the comparison set for the 2027–2028 procurement cycle. The chips that will be available in 2028 are being designed now, and the ones designed around CORDIC arithmetic will look substantially different from the ones designed without it.

The acceleration curve has a shape. It is visible in the data. The predictions above are an attempt to trace where that curve leads — not with certainty, but with the calibrated confidence that the evidence warrants.

---

## Go Deeper

The companion technical reference — [CORDIC in the AI Accelerator Stack](./CORDIC-in-the-AI-Accelerator-Stack.md) — provides the full architecture descriptions, silicon measurements, paper index, and mathematical foundations that ground every claim in this document.

The narrative introduction — [Just Shift and Add](./README-narrative.md) — provides the historical context and conceptual framing for readers approaching CORDIC for the first time.

---

*Predictions constructed from published results in IEEE TVLSI, IEEE TCAS-I/II, ACM TRETS, IEEE ARITH 2024, ISVLSI 2025, ISCAS 2025, VDAT 2025, VLSID 2026, and arXiv preprints through June 2026. Confidence levels represent the author's calibrated assessment of prediction accuracy given available evidence; they are not derived from formal probability models. All timelines are calendar year endpoints.*
