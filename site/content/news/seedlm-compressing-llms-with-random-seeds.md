+++
date = '2025-04-06T21:33:21+02:00'
draft = false
title = 'SeedLM: Compressing LLMs with Random Seeds'
+++

Researchers have introduced [SeedLM](https://arxiv.org/abs/2410.10714), a new post-training method that compresses large language models by encoding weights as seeds for pseudo-random generators. By reconstructing weights on the fly using Linear Feedback Shift Registers (LFSRs), SeedLM reduces memory access and speeds up inference, especially on memory-bound hardware like FPGAs. Unlike many compression methods, it is data-free and shows strong generalization across tasks. Tests on Llama3 70B achieve state-of-the-art 3-4 bit compression with zero-shot accuracy on par with FP16, and FPGA experiments show a 4× speed-up over FP16 baselines.