## Release Notes 
- [2025/07/23] 🔥 We release our preference data!
- [2025/07/22] 🔥 Our training code is under construction, stay tuned
- [2025/06/25] 🔥 Our paper has been accepted by ICCV 2025



# SPO

<p align="center">
  <img src="./docs/coverimage.jpg" width="500px"> <br>
</p>

We introduce structured policy optimization (SPO) -- a novel preference optimization method that simultaneously aligns preference instructions, responses, and dialogue interactions to improve multi-modal understanding and reasoning capabilities. The efficacy of SPO is attributed to one key design:
treating the questioning and answering as a sequential action and binding them through a trajectory reward. This reward formulation better aligns with real-world dialogue studies and eliminates the need for fixed instructions. 



## Models & Scripts

### Installation

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[train]"
pip install packaging &&  pip install ninja && pip install flash-attn --no-build-isolation --no-cache-dir # if flash-attention install error
```

## Training Data

### Video data: 
- [ShareGPTVideo](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction)

### Image data: 
- [[OneVision data preparation script]](https://github.com/heliossun/Structure-Policy-Optimization/blob/main/scripts/prepare_trainData/getData.py) 
- [m4 Instruct data](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data)

### Annotation data: 
- [stage-1 SFT data](https://huggingface.co/datasets/ZachSun/video-lvlm-data/blob/main/ours_interleave_iv.json) 
- [stage-2 SPO data](https://huggingface.co/datasets/ZachSun/video-lvlm-data/blob/main/merge_prefQA_7B_14500.json)





## Citation

If you find it useful for your research and applications, please cite related papers/blogs using this BibTeX:
```
@InProceedings{guohao_sun_ICCV,
    author    = {Sun, Guohao and Qin, Can and Yihao, Feng and Chen, Zeyuan and Xu, Ran and Dianat, Sohail and Rabbani, Majid and Rao, Raghuveer and Tao, Zhiqiang},
    title     = {Structured Policy Optimization: Enhance Large Vision-Language Model via Self-Referenced Dialogue},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
}
```

## Acknowledgement

- [LLaVA-NeXT](https://github.com/lm-sys/FastChat): the codebase we built upon!
- The LLaVA-NeXT project is currently maintained by the team: [Bo Li](https://brianboli.com/), [Dong Guo](https://www.linkedin.com/in/dongguoset/), [Feng Li](https://scholar.google.com/citations?hl=zh-CN&user=ybRe9GcAAAAJ&view_op=list_works&sortby=pubdate), [Hao Zhang](https://scholar.google.com/citations?user=B8hPxMQAAAAJ&hl=en), [Kaichen Zhang](https://www.linkedin.com/in/kaichen-zhang-014b17219/?originalSubdomain=sg), [Renrui Zhang](https://zrrskywalker.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/), led by [Chunyuan Li](https://chunyuan.li/) and with the guidance and help from [Haotian Liu](https://hliu.cc/).
- The `lmms-eval` is an easy to use inference framework, we mainly use it to evaluate our model's performance.

## Related Projects

- [SQ-LlaVA: Self-questioning for Vision-Language Assistant](https://github.com/heliossun/SQ-LLaVA)
- [STLLaVA-Med: Self-Training Large Language and Vision Assistant for Medical](https://github.com/heliossun/STLLaVA-Med)
