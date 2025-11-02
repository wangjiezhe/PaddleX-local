#!/usr/bin/env python

import gc
import io
import logging
import os  # noqa: F401
from pathlib import Path
from typing import Annotated, Optional

import img2pdf  # type: ignore
import typer

app = typer.Typer(help="Convert PDF and image files to Markdown using PaddleX PP-StructureV3")


class Colors:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    RESET = "\033[0m"


def release_gpu_memory():
    import paddle

    paddle.device.cuda.empty_cache()
    gc.collect()


def pil_to_pdf_img2pdf(pil_images, output_path: Path):
    """
    images2pdf
    """
    if not pil_images:
        return

    image_bytes_list = []

    for img in pil_images:
        if img.mode != "RGB":
            img = img.convert("RGB")

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)

    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        assert pdf_bytes is not None
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")


def process_image_file(
    image_path: Path, pipeline, output_dir: Path, save_layout=True, save_all=False
) -> Path:
    """
    处理单个图片文件，转换为 Markdown
    """
    typer.echo(f"🤖 Processing image file: {image_path}")

    ## 执行预测
    output = pipeline.predict(input=str(image_path))

    ## 生成输出文件路径
    mkd_file_path = output_dir / f"{image_path.stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    ## 保存当前图像的markdown格式的结果
    for res in output:
        if save_all:
            res.save_all(save_path=output_dir)
        else:
            if save_layout:
                res.save_to_img(save_path=output_dir)
            res.save_to_markdown(save_path=output_dir)
            res.save_to_xlsx(save_path=output_dir)

    return mkd_file_path


def process_pdf_file(
    pdf_path: Path,
    pipeline,
    output_dir: Path,
    v3=False,
    save_layout=True,
    save_all=False,
) -> Path:
    """
    处理 PDF 文件，转换为 Markdown
    """
    typer.echo(f"🤖 Processing PDF file: {pdf_path}")

    ## 执行预测
    if v3:
        output = pipeline.predict_iter(input=str(pdf_path), use_queues=True)
    else:
        output = pipeline.predict(input=str(pdf_path), use_queues=True)

    markdown_list = []
    markdown_images = []
    res_images = []

    ## This is needed for PaddleOCR-VL
    release_gpu_memory()

    for res in output:
        index = res.get("page_index") + 1
        typer.echo(f"🎲 parsing page {index} of file {pdf_path.name} ...")
        md_info = res.markdown
        markdown_list.append(md_info)
        if save_all:
            res.save_all(save_path=output_dir)
        else:
            if save_layout:
                res_images.append(res.img)
            res.save_to_xlsx(save_path=output_dir)
            markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    ## 生成输出文件路径
    mkd_file_path = output_dir / f"{pdf_path.stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    ## 写入Markdown文件
    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    if save_all:
        return mkd_file_path

    ## 保存可视化图像
    if save_layout:
        for layout in res_images[0].keys():
            # [
            #     "preprocessed_img",  # 预处理
            #     "layout_det_res",    # 显示版面区域检测
            #     "region_det_res",    # 区域检测（大块）
            #     "overall_ocr_res",   # OCR
            #     "layout_order_res",  # 显示顺序检测
            # ]:
            layout_pdf = output_dir / f"{pdf_path.stem}_{layout}.pdf"
            typer.echo(f"🚀 Saving {layout} results to: {layout_pdf}")
            pil_to_pdf_img2pdf([item[layout] for item in res_images], layout_pdf)

    typer.echo("🚀 Saving images in markdown")

    ## 保存图像
    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_dir / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)

    return mkd_file_path


@app.command()
def convert(
    input_files: Annotated[
        list[Path],
        typer.Argument(help="Input PDF or image file paths (multiple files supported)"),
    ],
    output_dir: Annotated[Path, typer.Option("-o", "--output", help="Output directory path")] = Path(
        "./output"
    ),
    hpip: Annotated[bool, typer.Option("--hpip", help="Enable high performance inference")] = False,
    config: Annotated[
        Optional[str],
        typer.Option("-c", "--config", help="PaddleX pipeline configuration"),
    ] = None,
    v3: Annotated[bool, typer.Option("--v3", help="Use PP-StructureV3 Pipeline")] = False,
    vl: Annotated[bool, typer.Option("--vl", help="Use PaddleOCR-VL Pipeline")] = False,
    no_layout: Annotated[bool, typer.Option("--no_layout", help="Do not save layout images")] = False,
    use_doc_unwarping: Annotated[
        bool,
        typer.Option("--use_doc_unwarping", help="Use the document unwarping module"),
    ] = False,
    use_doc_orientation_classify: Annotated[
        bool,
        typer.Option(
            "--use_doc_orientation_classify",
            help="Use the document orientation classification module",
        ),
    ] = False,
    use_textline_orientation: Annotated[
        bool,
        typer.Option(
            "--use_textline_orientation",
            help="Use the text line orientation classification",
        ),
    ] = False,
    use_table_recognition: Annotated[
        bool,
        typer.Option("--use_table_recognition", help="Use table recognition subpipeline"),
    ] = False,
    use_chart_recognition: Annotated[
        bool,
        typer.Option("--use_chart_recognition", help="Use the chart parsing module"),
    ] = False,
    save_all: Annotated[bool, typer.Option("--save_all", help="Save all results directly")] = False,
):
    """
    Convert PDF and image files to Markdown format.

    Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif

    Examples:
        python pdf2md.py file1.pdf
        python pdf2md.py file1.pdf file2.pdf file3.jpg
        python pdf2md.py *.pdf -o ./output --v3
    """
    if not input_files:
        typer.echo("Error: No input files provided", err=True)
        raise typer.Exit(code=1)

    ## 支持的图片格式
    ## 来自 paddlex 里的
    ## inference/common/batch_sampler/image_batch_sampler.py
    supported_image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    ## 检查所有输入文件
    for input_file in input_files:
        if not input_file.exists():
            typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
            raise typer.Exit(code=1)

        file_extension = input_file.suffix.lower()
        if file_extension not in {".pdf"} | supported_image_extensions:
            typer.echo(
                f"Error: Unsupported file format '{file_extension}' for file '{input_file}'. "
                f"Supported formats: PDF, {', '.join(supported_image_extensions)}",
                err=True,
            )
            raise typer.Exit(code=1)

    if len(input_files) == 1:
        typer.echo("▶️ Processing 1 file...")
    else:
        typer.echo(f"▶️ Processing {len(input_files)} files...")

    if hpip:
        if v3 or vl:
            typer.echo(
                f"{Colors.RED}`--hpip` does not work with `--v3` and `--vl`. Ignored.",
                err=True,
                color=True,
            )
        typer.echo("🚀 Enabling high performance inference mode")

    if config and (v3 or vl):
        typer.echo(
            f"{Colors.RED}`--v3` and `--vl` does not work with `--config`. Ignored.",
            err=True,
            color=True,
        )
        vl = False
        v3 = False

    ## 初始化流水线
    if vl:
        ## 必须在引入任何 paddle 模块前设置才能够起作用
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        from paddleocr import PaddleOCRVL  # type: ignore

        pipeline = PaddleOCRVL(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
        )
    elif v3:
        from paddleocr import PPStructureV3

        pipeline = PPStructureV3(
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_table_recognition=use_table_recognition,
            use_chart_recognition=use_chart_recognition,
        )
    else:
        from paddlex import create_pipeline  # type: ignore

        ## 创建PP-StructureV3流水线
        pipeline_config = config or "./PP-StructureV3-notable.yaml"
        pipeline = create_pipeline(
            pipeline=pipeline_config,
            use_hpip=hpip,
            hpi_config={"auto_config": "False", "backend": "onnxruntime"},
        )

    logging.getLogger("paddlex").setLevel(logging.ERROR)

    ## 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    ## 处理所有文件
    successful_conversions = []
    for index, input_file in enumerate(input_files, 1):
        file_extension = input_file.suffix.lower()
        typer.echo(f"\n--- Processing file {index}/{len(input_files)} ---")

        try:
            ## 根据文件类型处理
            if file_extension == ".pdf":
                output_path = process_pdf_file(
                    input_file,
                    pipeline,
                    output_dir,
                    v3=v3 or vl,
                    save_layout=not no_layout,
                    save_all=save_all,
                )
            else:
                output_path = process_image_file(
                    input_file,
                    pipeline,
                    output_dir,
                    save_layout=not no_layout,
                    save_all=save_all,
                )
            successful_conversions.append(output_path)
            typer.echo(f"✅ File {index} conversion completed! Markdown file saved to: {output_path}")
        except Exception as e:
            typer.echo(f"❌ Error processing file {index} ({input_file}): {str(e)}", err=True)
            continue

    ## 总结结果
    typer.echo("\n🎉 Batch conversion completed!")
    typer.echo(f"Successfully converted {len(successful_conversions)} out of {len(input_files)} files")

    if successful_conversions:
        typer.echo("Output files:")
        for output_path in successful_conversions:
            typer.echo(f"  - {output_path}")


if __name__ == "__main__":
    app()
