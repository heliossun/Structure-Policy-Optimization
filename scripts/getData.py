import os
from datasets import load_dataset
from tqdm import tqdm
import json


subfolder=['figureqa(cauldron,llava_format)', 'mathqa', 'PMC-VQA(MathV360K)', 'vqarad(cauldron,llava_format)', 'Geometry3K(MathV360K)', 'infographic_vqa', 'multihiertt(cauldron)', 'ai2d(cauldron,llava_format)', 'geomverse(cauldron)', 'tabmwp(cauldron)', 'hateful_memes(cauldron,llava_format)', 'infographic_vqa_llava_format', 'vsr(cauldron,llava_format)', 'iam(cauldron)', 'scienceqa(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'textocr(gpt4v)', 'hme100k', 'IconQA(MathV360K)', 'geo170k(align)', 'mavis_math_rule_geo', 'GeoQA+(MathV360K)', 'textcaps', 'ai2d(internvl)', 'mavis_math_metagen', 'sharegpt4v(sam)', 'vistext(cauldron)', 'llavar_gpt4_20k', 'sroie', 'lrv_normal(filtered)', 'geo170k(qa)', 'rendered_text(cauldron)', 'clevr(cauldron,llava_format)', 'robut_wtq(cauldron,llava_format)', 'magpie_pro(qwen2_72b_st)', '.gitattributes', 'sharegpt4v(coco)', 'chrome_writting', 'sharegpt4v(knowledge)', '.git', 'robut_sqa(cauldron)', 'MapQA(MathV360K)', 'tqa(cauldron,llava_format)', 'tallyqa(cauldron,llava_format)', 'orand_car_a', 'aokvqa(cauldron,llava_format)', 'lrv_chart', 'visualmrc(cauldron)', 'image_textualization(filtered)', 'mapqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'ai2d(gpt4v)', 'sharegpt4o', 'intergps(cauldron,llava_format)', 'robut_wikisql(cauldron)', 'UniGeo(MathV360K)', 'chartqa(cauldron,llava_format)', 'VizWiz(MathV360K)', 'dvqa(cauldron,llava_format)', 'magpie_pro(l3_80b_st)', 'infographic(gpt4v)', 'FigureQA(MathV360K)', 'websight(cauldron)', 'visual7w(cauldron,llava_format)', 'CLEVR-Math(MathV360K)', 'k12_printing', 'GEOS(MathV360K)', 'st_vqa(cauldron,llava_format)', 'raven(cauldron)', 'sharegpt4v(llava)', 'geo3k', 'chart2text(cauldron)', 'iiit5k', 'screen2words(cauldron)', 'vision_flan(filtered)', 'diagram_image_to_text(cauldron)', 'TabMWP(MathV360K)', 'magpie_pro(l3_80b_mt)', 'allava_instruct_vflan4v', 'Super-CLEVR(MathV360K)', 'allava_instruct_laion4v', 'iconqa(cauldron,llava_format)']


image_folder = "/home/gs4288/guohao/data/onevision/images"

converted_data = []
for subf in subfolder:
    data = load_dataset("lmms-lab/LLaVA-OneVision-Data", subf,split="train")
    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            json_data["image"] = f"{da['id']}.jpg"
            da["image"].save(os.path.join(image_folder, json_data["image"]))
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)


with open("/home/gs4288/guohao/data/onevision/ov_instruct.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)

['CLEVR-Math(MathV360K)', 'FigureQA(MathV360K)', 'GEOS(MathV360K)', 'GeoQA+(MathV360K)', 'Geometry3K(MathV360K)', 'IconQA(MathV360K)', 'MapQA(MathV360K)', 'PMC-VQA(MathV360K)', 'Super-CLEVR(MathV360K)', 'TabMWP(MathV360K)', 'UniGeo(MathV360K)', 'VizWiz(MathV360K)', 'ai2d(cauldron,llava_format)', 'ai2d(gpt4v)', 'ai2d(internvl)', 'allava_instruct_laion4v', 'allava_instruct_vflan4v', 'aokvqa(cauldron,llava_format)', 'chart2text(cauldron)', 'chartqa(cauldron,llava_format)', 'chrome_writting', 'clevr(cauldron,llava_format)', 'diagram_image_to_text(cauldron)', 'dvqa(cauldron,llava_format)', 'figureqa(cauldron,llava_format)', 'geo170k(align)', 'geo170k(qa)', 'geo3k', 'geomverse(cauldron)', 'hateful_memes(cauldron,llava_format)', 'hitab(cauldron,llava_format)', 'hme100k', 'iam(cauldron)', 'iconqa(cauldron,llava_format)', 'iiit5k', 'image_textualization(filtered)', 'infographic(gpt4v)', 'infographic_vqa', 'infographic_vqa_llava_format', 'intergps(cauldron,llava_format)', 'k12_printing', 'llavar_gpt4_20k', 'lrv_chart', 'lrv_normal(filtered)', 'magpie_pro(l3_80b_mt)', 'magpie_pro(l3_80b_st)', 'magpie_pro(qwen2_72b_st)', 'mapqa(cauldron,llava_format)', 'mathqa', 'mavis_math_metagen', 'mavis_math_rule_geo', 'multihiertt(cauldron)', 'orand_car_a', 'raven(cauldron)', 'rendered_text(cauldron)', 'robut_sqa(cauldron)', 'robut_wikisql(cauldron)', 'robut_wtq(cauldron,llava_format)', 'scienceqa(cauldron,llava_format)', 'scienceqa(nona_context)', 'screen2words(cauldron)', 'sharegpt4o', 'sharegpt4v(coco)', 'sharegpt4v(knowledge)', 'sharegpt4v(llava)', 'sharegpt4v(sam)', 'sroie', 'st_vqa(cauldron,llava_format)', 'tabmwp(cauldron)', 'tallyqa(cauldron,llava_format)', 'textcaps', 'textocr(gpt4v)', 'tqa(cauldron,llava_format)', 'vision_flan(filtered)', 'vistext(cauldron)', 'visual7w(cauldron,llava_format)', 'visualmrc(cauldron)', 'vqarad(cauldron,llava_format)', 'vsr(cauldron,llava_format)', 'websight(cauldron)']