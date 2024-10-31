## Release Notes

- [2024/08/15] 🔥 


## Experiment Notes

- [x] Support training from llava-ov and LoRA fine-tuning.
- [x] Support self-question training.
- [] Support token merging ([ToMe](https://arxiv.org/abs/2210.09461)).
- [] Support our DPO
# LLaVA-NeXT: Open Large Multimodal Models
[![llava_onevision-checkpoints](https://img.shields.io/badge/llava_onevision-checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-onevision-66a259c3526e15166d6bba37)
[![llava_next-interleave_checkpoints](https://img.shields.io/badge/llava_next-interleave_checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-next-interleave-66763c55c411b340b35873d1)
[![llava_next-video_checkpoints](https://img.shields.io/badge/llava_next-video_checkpoints-blue)](https://huggingface.co/collections/lmms-lab/llava-next-video-661e86f5e8dabc3ff793c944)
[![llava_next-image_checkpoints](https://img.shields.io/badge/llava_next-image_checkpoints-blue)](https://huggingface.co/lmms-lab)




## Models & Scripts

### Installation

#### 1. **Clone this repository and navigate to the LLaVA folder:**
```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
```

#### 2. **Install the inference package:**
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

- [Vicuna](https://github.com/lm-sys/FastChat): the codebase we built upon, and our base model Vicuna-13B that has the amazing language capabilities!
- The LLaVA-NeXT project is currently maintained by the team along with our contributors (listed alphabetically by the first names): [Bo Li](https://brianboli.com/), [Dong Guo](https://www.linkedin.com/in/dongguoset/), [Feng Li](https://scholar.google.com/citations?hl=zh-CN&user=ybRe9GcAAAAJ&view_op=list_works&sortby=pubdate), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en), [Kaichen Zhang](https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg), [Renrui Zhang](https://zrrskywalker.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/), led by [Chunyuan Li](https://chunyuan.li/) and with the guidance and help from [Haotian Liu](https://hliu.cc/).
- The `﻿lmms-eval` framework and its core contributors, including Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, and Kairui Hu, for their support on the evaluation side.

## Related Projects

- [Instruction Tuning with GPT-4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://github.com/microsoft/LLaVA-Med)
- [Otter: In-Context Multi-Modal Instruction Tuning](https://github.com/Luodian/Otter)

For future project ideas, please check out:
- [SEEM: Segment Everything Everywhere All at Once](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) to detect, segment, and generate anything by marrying [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-Anything](https://github.com/facebookresearch/segment-anything).
