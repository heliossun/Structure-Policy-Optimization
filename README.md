## Release Notes
- [2025/07/22] 🔥 Our training code was released!
- [2025/06/25] 🔥 Our paper has been accepted by ICCV 2025



# SPO
[![SPO-llava-onevision checkpoints](https://img.shields.io/badge/llava_onevision-checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37)





## Models & Scripts

### Installation

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir # if flash-attention install error
```





## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```bibtex

```

## Acknowledgement

- [LLaVA-NeXT](https://github.com/lm-sys/FastChat): the codebase we built upon!
- The LLaVA-NeXT project is currently maintained by the team: [Bo Li](https://brianboli.com/), [Dong Guo](https://www.linkedin.com/in/dongguoset/), [Feng Li](https://scholar.google.com/citations?hl=zh-CN&user=ybRe9GcAAAAJ&view_op=list_works&sortby=pubdate), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en), [Kaichen Zhang](https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg), [Renrui Zhang](https://zrrskywalker.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/), led by [Chunyuan Li](https://chunyuan.li/) and with the guidance and help from [Haotian Liu](https://hliu.cc/).
- The `﻿lmms-eval` is an easy to use inference framework, we mainly use it to evaluate our model's performance.

## Related Projects

- [SQ-LlaVA: Self-questioning for Vision-Language Assistant](https://github.com/heliossun/SQ-LLaVA)
- [STLLaVA-Med: Self-Training Large Language and Vision Assistant for Medical](https://github.com/heliossun/STLLaVA-Med)
