# PaddleX

## 构建环境

```bash
# uv venv --seed --python python3.11 --system-site-packages
uv venv --seed --python python3.12
uv pip install paddlepaddle-gpu==3.2.0 --default-index https://www.paddlepaddle.org.cn/packages/stable/cu129/
uv pip install "paddleocr[all]" --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install ipython typer --default-index https://pypi.tuna.tsinghua.edu.cn/simple
export LD_LIBRARY_PATH=/usr/lib64:/opt/cuda/lib64:/usr/lib/wsl/lib:/usr/lib
```

##  PaddleOCR 产线与 PaddleX 产线注册名的对应关系

| PaddleOCR 产线       | PaddleX 产线注册名     |
|:--------------------:|:----------------------:|
| 通用 OCR             | OCR                    |
| PP-StructureV3       | PP-StructureV3         |
| PP-ChatOCRv4         | PP-ChatOCRv4-doc       |
| 通用表格识别 v2      | table_recognition_v2   |
| 公式识别             | formula_recognition    |
| 印章文本识别         | seal_recognition       |
| 文档图像预处理       | doc_preprocessor       |
| 文档理解             | doc_understanding      |
| PP-DocTranslation    | PP-DocTranslation      |

## 注意事项

- 如果图片是没有歪扭，建议关掉`use_doc_unwarping`，否则可能把正常的图片变得歪扭，还会进行不必要的裁剪影响正常的文字识别。

## 高性能推理

```bash
uv run paddleocr install_hpi_deps gpu
```

按照[官方的回复](https://github.com/PaddlePaddle/PaddleX/issues/4336#issuecomment-3049637910)，目前最新版本的paddle框架还不支持自动配置模式，需要开启手动配置模式：

```bash
uv run paddlex \
    --pipeline ./PP-StructureV3-nowarping.yaml \
    --input ./input/course.jpg \
    --save_path ./output
    --use_hpip \
    --hpi_config '{"auto_config": "False", "backend": "onnxruntime"}'
```

不过开启高性能推理的代价是占用显存变多。由于我的本地的显存太小，开启高性能推理后反而变慢，甚至有时会报错。


## 使用支持TensorRT的镜像

PaddleX的后端推理库（ultra_infer）目前只支持TensorRT-8，而其又依赖CUDNN-8。因此选择直接使用容器。

启动容器：

```bash
docker run --gpus all --name paddlex -v $PWD:/paddle -v /root/.paddleocr:/root/.paddleocr -v /root/.paddlex:/root/.paddlex --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.3.4-paddlepaddle3.2.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash
```

容器内部：

```bash
paddlex --install hpi-gpu

cd /paddle
paddlex \
    --pipeline PP-StructureV3 \
    --input ./input/course.jpg \
    --save_path ./output \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --use_seal_recognition False \
    --use_table_recognition True \
    --use_hpip \
    --hpi_config '{"auto_config": "False", "backend": "tensorrt", "backend_config": {"precision": "fp16"}}'
```

使用TensorRT要求显存必须足够，不能使用内存，否则会报错。

paddlex支持的参数：

```
usage: Command-line interface for PaddleX. Use the options below to install plugins, run pipeline predictions, or start the serving application.
       [-h] [--install [PLUGIN ...]] [--no_deps] [--platform {github.com,gitee.com}] [-y] [--use_local_repos] [--deps_to_replace DEPS_TO_REPLACE [DEPS_TO_REPLACE ...]]
       [--pipeline PIPELINE] [--input INPUT] [--save_path SAVE_PATH] [--device DEVICE] [--use_hpip] [--hpi_config HPI_CONFIG] [--get_pipeline_config GET_PIPELINE_CONFIG]
       [--serve] [--host HOST] [--port PORT] [--paddle2onnx] [--paddle_model_dir PADDLE_MODEL_DIR] [--onnx_model_dir ONNX_MODEL_DIR] [--opset_version OPSET_VERSION]
       [--use_doc_orientation_classify USE_DOC_ORIENTATION_CLASSIFY] [--use_doc_unwarping USE_DOC_UNWARPING] [--use_general_ocr USE_GENERAL_OCR]
       [--use_textline_orientation USE_TEXTLINE_ORIENTATION] [--use_seal_recognition USE_SEAL_RECOGNITION] [--use_table_recognition USE_TABLE_RECOGNITION]
       [--use_formula_recognition USE_FORMULA_RECOGNITION] [--layout_threshold LAYOUT_THRESHOLD] [--layout_nms LAYOUT_NMS] [--layout_unclip_ratio LAYOUT_UNCLIP_RATIO]
       [--layout_merge_bboxes_mode LAYOUT_MERGE_BBOXES_MODE] [--seal_det_limit_side_len SEAL_DET_LIMIT_SIDE_LEN] [--seal_det_limit_type SEAL_DET_LIMIT_TYPE]
       [--seal_det_thresh SEAL_DET_THRESH] [--seal_det_box_thresh SEAL_DET_BOX_THRESH] [--seal_det_unclip_ratio SEAL_DET_UNCLIP_RATIO]
       [--seal_rec_score_thresh SEAL_REC_SCORE_THRESH] [--text_det_limit_side_len TEXT_DET_LIMIT_SIDE_LEN] [--text_det_limit_type TEXT_DET_LIMIT_TYPE]
       [--text_det_thresh TEXT_DET_THRESH] [--text_det_box_thresh TEXT_DET_BOX_THRESH] [--text_det_unclip_ratio TEXT_DET_UNCLIP_RATIO]
       [--text_rec_score_thresh TEXT_REC_SCORE_THRESH] [--use_table_cells_ocr_results USE_TABLE_CELLS_OCR_RESULTS]
       [--use_e2e_wired_table_rec_model USE_E2E_WIRED_TABLE_REC_MODEL] [--use_e2e_wireless_table_rec_model USE_E2E_WIRELESS_TABLE_REC_MODEL]
```

## 使用PaddleOCR-VL

- 需要安装最新版本的`paddlex==3.3.6`，3.3.5本地运行失败。
- 需要安装特殊版本的`safetensors`。

```bash
uv pip install paddlex==3.3.6
uv pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```

```bash
uv run paddleocr doc_parser -i input/ch4.pdf \
    --save_path output \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_chart_recognition False
```

```bash
uv run python pdf2md.py input/ch4.pdf -o output --vl
```

使用容器：

```bash
docker run \
    -it \
    --name paddleocr-vl \
    -v $PWD:/paddle -v /root/.paddleocr:/home/paddleocr/.paddleocr -v /root/.paddlex:/home/paddleocr/.paddlex \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest \
    /bin/bash
```
