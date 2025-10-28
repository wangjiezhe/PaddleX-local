# PaddleX

## 构建环境

```bash
uv venv --seed --python python3.12
source .venv/bin/activate
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
pip install "paddleocr[all]" -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install setuptools ipython -i https://pypi.tuna.tsinghua.edu.cn/simple
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
