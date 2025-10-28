# PaddleX

## 构建环境

```bash
uv venv --seed --python python3.11 --system-site-packages
uv pip install paddlepaddle-gpu==3.2.0 --default-index https://www.paddlepaddle.org.cn/packages/stable/cu129/
uv pip install "paddleocr[all]" --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install setuptools ipython typer --default-index https://pypi.tuna.tsinghua.edu.cn/simple
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

按照[官方的回复](https://github.com/PaddlePaddle/PaddleX/issues/4336#issuecomment-3049637910)，目前最新版本的paddle框架还不支持自动配置模式，需要开机手动配置模式：

```bash
uv run paddlex \
    --pipeline ./PP-StructureV3-nowarping.yaml \
    --input test.jpg \
    --use_hpip \
    --hpi_config '{"auto_config": "False", "backend": "onnxruntime"}'
```

不过开启高性能推理的代价是占用显存变多。由于我的本地的显存台下，开启高性能推理后反而变慢，设置有时会报错。
