#!/usr/bin/env python

from pathlib import Path

import typer

app = typer.Typer(
    help="Convert PDF and image files to Markdown using PaddleX PP-StructureV3"
)


def process_image_file(image_path: Path, pipeline, output_dir: Path) -> Path:
    """
    处理单个图片文件，转换为 Markdown
    """
    typer.echo(f"Processing image file: {image_path}")

    # 执行预测
    output = pipeline.predict(input=str(image_path))

    # 生成输出文件路径
    mkd_file_path = output_dir / f"{image_path.stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存当前图像的markdown格式的结果
    for res in output:
        res.save_to_markdown(save_path=output_dir)

    return mkd_file_path


def process_pdf_file(pdf_path: Path, pipeline, output_dir: Path) -> Path:
    """
    处理 PDF 文件，转换为 Markdown
    """
    typer.echo(f"Processing PDF file: {pdf_path}")

    # 执行预测
    output = pipeline.predict(input=str(pdf_path))

    markdown_list = []
    markdown_images = []

    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    # 生成输出文件路径
    mkd_file_path = output_dir / f"{pdf_path.stem}.md"
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入Markdown文件
    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    # 保存图像
    for item in markdown_images:
        if item:
            for path, image in item.items():
                file_path = output_dir / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(file_path)

    return mkd_file_path


@app.command()
def convert(
    input_file: Path = typer.Argument(..., help="Input PDF or image file path"),
    output_dir: Path = typer.Option(
        "./output", "-o", "--output", help="Output directory path"
    ),
    hpip: bool = typer.Option(
        False, "--hpip", help="Enable high performance inference"
    ),
    vl: bool = typer.Option(False, "--vl", help="Use PaddleOCR-VL model"),
    config: str = typer.Option(
        None, "-c", "--config", help="PaddleX pipeline configuration"
    ),
):
    """
    Convert PDF and image files to Markdown format.

    Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    # 检查输入文件是否存在
    if not input_file.exists():
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(code=1)

    # 支持的图片格式
    supported_image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    file_extension = input_file.suffix.lower()

    # 验证文件格式
    if file_extension not in {".pdf"} | supported_image_extensions:
        typer.echo(
            f"Error: Unsupported file format '{file_extension}'. "
            f"Supported formats: PDF, {', '.join(supported_image_extensions)}",
            err=True,
        )
        raise typer.Exit(code=1)

    if hpip:
        typer.echo("🚀 Enabling high performance inference mode")

    if vl:
        from paddleocr import PaddleOCRVL  # type: ignore

        pipeline = PaddleOCRVL()
    else:
        from paddlex import create_pipeline  # type: ignore

        # 创建PP-StructureV3流水线
        pipeline_config = config or "./PP-StructureV3-notable.yaml"
        pipeline = create_pipeline(
            pipeline=pipeline_config,
            use_hpip=hpip,
            hpi_config={"auto_config": "False", "backend": "onnxruntime"},
        )

    # 根据文件类型处理
    if file_extension == ".pdf":
        output_path = process_pdf_file(input_file, pipeline, output_dir)
    else:
        output_path = process_image_file(input_file, pipeline, output_dir)

    typer.echo(f"✅ Conversion completed! Markdown file saved to: {output_path}")


if __name__ == "__main__":
    app()
