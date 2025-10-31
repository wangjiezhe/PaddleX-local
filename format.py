#!/usr/bin/env python

import re
from pathlib import Path

import typer

app = typer.Typer()


def parse_multiline_formula(match: re.Match[str]) -> str:
    block = match.group(1)

    formulas = re.split(r"\\\] \\\[", block)
    if len(formulas) == 1:
        return match.group(0)

    res = r"\[\begin{aligned}"
    res += "\n"
    for i, formula in enumerate(formulas):
        res += f"&{formula}"
        if i < len(formulas) - 1:
            res += r"\\"
        res += "\n"
    res += r"\end{aligned}\]"
    return res


def parse_parallel(match: re.Match[str]) -> str:
    res = match.group(2)
    res = re.sub(r"/\s*/", r"\\parallel", res)
    return match.group(1) + res + match.group(3)


def format_deepseek(content: str) -> str:
    """
    Format markdown file OCR by DeepSeek-OCR
    """
    # 去掉换页标记
    content = content.replace("<--- Page Split --->\n", "")
    # 替换识别错误的乘号
    content = content.replace(r"\bullet", r"\cdot")
    # 替换平行符号
    content = re.sub(r"(\\\()(.*?)(\\\))", parse_parallel, content)
    content = re.sub(r"(\\\[)(.*?)(\\\])", parse_parallel, content)
    content = re.sub(
        r"\\\((.*?)\\\) // \\\((.*?)\\\)", r"\(\1 \\parallel \2\)", content
    )
    # 合并多行公式
    content = re.sub(r"\\\[(.*)\\\]", parse_multiline_formula, content)
    # 正确显示多行公式
    content = re.sub(r"\\\[(.*?)\\\]", r"\[\n\1\n\]", content, flags=re.DOTALL)
    return content


def format_paddle(content: str) -> str:
    """
    Format markdown file OCR by PaddleOCR
    """
    # 替换识别错误的乘号
    content = content.replace(r"\bullet", r"\cdot")
    # 替换平行符号
    content = re.sub(r"(\$)(.+?)(\$)", parse_parallel, content)
    # 正确显示多行公式
    content = re.sub(r"\$\$(.+?)\$\$", r"$$\n\1\n$$", content)
    return content


@app.command()
def main(
    input_file: Path = typer.Argument(..., help="Input markdown file"),
    formatter: str = typer.Option(
        "deepseek", "-f", "--formatter", help="Formatter type: 'deepseek' or 'paddle'"
    ),
):
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    if formatter == "deepseek":
        content = format_deepseek(content)
    elif formatter == "paddle":
        content = format_paddle(content)
    else:
        raise typer.BadParameter("Formatter must be either 'deepseek' or 'paddle'")

    with open(
        input_file.parent / f"{input_file.stem}_modified{input_file.suffix}",
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        _ = f.write(content)


if __name__ == "__main__":
    app()
