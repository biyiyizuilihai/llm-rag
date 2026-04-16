import os
import io
import json
import math
import base64
import argparse
from typing import List, Tuple, Optional

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI


DEFAULT_MODEL = "qwen3.5-35b-a3b"
# 来自 Qwen 官方模型卡：默认上下文 262,144
MODEL_CONTEXT_WINDOWS = {
    "qwen3.5-35b-a3b": 262_144,
}


def resize_image_if_needed(img: Image.Image, max_pixels: Optional[int]) -> Image.Image:
    """按总像素上限缩放，减少请求体大小和视觉 token。"""
    if not max_pixels:
        return img

    w, h = img.size
    pixels = w * h
    if pixels <= max_pixels:
        return img

    scale = math.sqrt(max_pixels / pixels)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def pil_to_data_url(img: Image.Image, jpeg_quality: int = 80) -> str:
    """转成 data:image/jpeg;base64,..."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def render_pdf_pages(
    pdf_path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    dpi: int = 144,
    local_max_pixels: Optional[int] = 1_500_000,
    jpeg_quality: int = 80,
) -> List[Tuple[int, str]]:
    """
    把 PDF 渲染成 [(页码, data_url), ...]
    start_page / end_page 为 1-based, 且 end_page 包含在内。
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if end_page is None:
        end_page = total_pages

    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(
            f"页码范围无效：start_page={start_page}, end_page={end_page}, total_pages={total_pages}"
        )

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    result: List[Tuple[int, str]] = []
    for page_num in range(start_page, end_page + 1):
        page = doc.load_page(page_num - 1)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = resize_image_if_needed(img, local_max_pixels)
        data_url = pil_to_data_url(img, jpeg_quality=jpeg_quality)
        result.append((page_num, data_url))

    doc.close()
    return result


def build_messages(
    page_images: List[Tuple[int, str]],
    question: str,
    api_max_pixels_per_page: Optional[int] = 1_048_576,
):
    """
    构造 OpenAI-compatible Chat Completions messages。
    每页前面加一个页码说明，方便模型在回答里引用页码。
    """
    intro = (
        "你将看到同一个 PDF 的连续页面图片。"
        "请结合所有页面回答最后的问题。"
        "如果答案依赖具体页面，请尽量注明页码。"
        "若信息不足，请明确说缺少哪部分。"
    )

    content = [{"type": "text", "text": intro}]

    for page_num, data_url in page_images:
        content.append({"type": "text", "text": f"下面是 PDF 第 {page_num} 页。"})
        image_item = {
            "type": "image_url",
            "image_url": {"url": data_url},
        }
        # 百炼兼容接口支持给图像项传 max_pixels 控制视觉 token 成本
        if api_max_pixels_per_page:
            image_item["max_pixels"] = api_max_pixels_per_page
        content.append(image_item)

    content.append({"type": "text", "text": f"问题：{question}"})

    return [{"role": "user", "content": content}]


def usage_to_dict(usage_obj):
    if usage_obj is None:
        return {}
    if isinstance(usage_obj, dict):
        return usage_obj
    if hasattr(usage_obj, "model_dump"):
        return usage_obj.model_dump()
    # 兜底
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
        "completion_tokens": getattr(usage_obj, "completion_tokens", None),
        "total_tokens": getattr(usage_obj, "total_tokens", None),
        "prompt_tokens_details": getattr(usage_obj, "prompt_tokens_details", None),
    }


def ask_qwen_with_pdf_images(
    pdf_path: str,
    question: str,
    model: str = DEFAULT_MODEL,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    start_page: int = 1,
    end_page: Optional[int] = None,
    dpi: int = 144,
    local_max_pixels: Optional[int] = 1_500_000,
    api_max_pixels_per_page: Optional[int] = 1_048_576,
    jpeg_quality: int = 80,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    enable_thinking: bool = False,
):
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("请先设置环境变量 DASHSCOPE_API_KEY")

    page_images = render_pdf_pages(
        pdf_path=pdf_path,
        start_page=start_page,
        end_page=end_page,
        dpi=dpi,
        local_max_pixels=local_max_pixels,
        jpeg_quality=jpeg_quality,
    )

    messages = build_messages(
        page_images=page_images,
        question=question,
        api_max_pixels_per_page=api_max_pixels_per_page,
    )

    client = OpenAI(api_key=api_key, base_url=base_url)

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={
            "enable_thinking": enable_thinking
        },
    )

    answer_parts: List[str] = []
    final_usage = None

    print("\n========== 模型回答 ==========\n")
    for chunk in stream:
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                print(text, end="", flush=True)
                answer_parts.append(text)

        if getattr(chunk, "usage", None):
            final_usage = usage_to_dict(chunk.usage)

    print("\n\n========== 用量统计 ==========\n")

    context_window = MODEL_CONTEXT_WINDOWS.get(model)
    prompt_tokens = final_usage.get("prompt_tokens")
    completion_tokens = final_usage.get("completion_tokens")
    total_tokens = final_usage.get("total_tokens")

    prompt_details = final_usage.get("prompt_tokens_details") or {}
    cached_tokens = None
    if isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")

    # 两种“剩余”的算法
    remaining_before_generation = None
    if context_window is not None and prompt_tokens is not None:
        remaining_before_generation = context_window - prompt_tokens - max_tokens

    remaining_after_round = None
    if context_window is not None and total_tokens is not None:
        remaining_after_round = context_window - total_tokens

    summary = {
        "model": model,
        "context_window": context_window,
        "pdf_path": os.path.abspath(pdf_path),
        "pages_sent": [page_num for page_num, _ in page_images],
        "page_count_sent": len(page_images),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cached_tokens": cached_tokens,
        "requested_max_tokens": max_tokens,
        "remaining_before_generation_conservative": remaining_before_generation,
        "remaining_after_round": remaining_after_round,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return {
        "answer": "".join(answer_parts),
        "usage": final_usage,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(
        description="本地 PDF -> 页面图片 -> Qwen API -> 回答 + Token统计"
    )
    parser.add_argument("--pdf", required=True, help="PDF 文件路径")
    parser.add_argument("--question", required=True, help="你要问模型的问题")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="百炼 OpenAI 兼容接口 base_url")
    parser.add_argument("--start-page", type=int, default=1, help="起始页（1-based）")
    parser.add_argument("--end-page", type=int, default=None, help="结束页（1-based，包含）")
    parser.add_argument("--dpi", type=int, default=144, help="PDF 渲染 DPI")
    parser.add_argument("--local-max-pixels", type=int, default=1_500_000, help="本地缩放后的单页总像素上限")
    parser.add_argument("--api-max-pixels-per-page", type=int, default=1_048_576, help="传给接口的单页 max_pixels")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG 压缩质量")
    parser.add_argument("--max-tokens", type=int, default=2048, help="本次最多生成多少 token")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")
    parser.add_argument("--enable-thinking", action="store_true", help="开启 thinking；默认关闭以省 token")

    args = parser.parse_args()

    ask_qwen_with_pdf_images(
        pdf_path=args.pdf,
        question=args.question,
        model=args.model,
        base_url=args.base_url,
        start_page=args.start_page,
        end_page=args.end_page,
        dpi=args.dpi,
        local_max_pixels=args.local_max_pixels,
        api_max_pixels_per_page=args.api_max_pixels_per_page,
        jpeg_quality=args.jpeg_quality,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
    )


if __name__ == "__main__":
    main()