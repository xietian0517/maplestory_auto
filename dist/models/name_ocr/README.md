# 本地名字 OCR 模型

来自 [RapidOCR 3.9.2](https://pypi.org/project/rapidocr/3.9.2/) 发布包，使用 ONNX Runtime 在本机识别，不上传截图。源码及 Windows 发行包均使用 `models/name_ocr/` 相对路径。

- `PP-OCRv6_det_small.onnx`：检测文字区域。
- `PP-OCRv6_rec_small.onnx`：读取中英文文字。
- `ch_ppocr_mobile_v2.0_cls_mobile.onnx`：RapidOCR 初始化依赖的方向分类模型，本程序调用时关闭方向分类。

上游：[RapidOCR](https://github.com/RapidAI/RapidOCR)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)。随附上游 Apache-2.0 许可证；`SHA256SUMS.txt` 记录随包模型的校验值。
程序启动 OCR 前会检查三个文件是否存在，不在运行时自动下载。输入框留空时不加载 OCR 模型。
