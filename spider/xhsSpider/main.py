import json
import os
import re
import requests
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from loguru import logger
from apis.xhs_pc_apis import XHS_Apis
from dotenv import load_dotenv
from xhs_utils.common_util import init
from xhs_utils.data_util import handle_note_info

from paddleocr import PaddleOCR

ocr_engine = PaddleOCR(
    lang="ch",
    use_angle_cls=True,
    show_log=False,
    rec_model_dir="ch_PP-OCRv4_rec",
    det_model_dir="ch_PP-OCRv4_det"
)

_ocr_lock = threading.Lock()

# 长图最大高度限制（像素），超过此值会按比例缩放
MAX_IMAGE_HEIGHT = 4096
MAX_IMAGE_WIDTH = 2048

NOTE_WORKERS = 1


def _resize_if_needed(img_path):
    """
    检查图片尺寸，超大图（长图/宽图）按比例缩放后覆盖保存。
    防止 PaddleOCR 处理超大图时内存溢出。
    返回实际用于OCR的路径。
    """
    try:
        img = Image.open(img_path)
        w, h = img.size
        if h <= MAX_IMAGE_HEIGHT and w <= MAX_IMAGE_WIDTH:
            return img_path

        # 计算缩放比例
        scale = min(MAX_IMAGE_WIDTH / w, MAX_IMAGE_HEIGHT / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        logger.debug(f"  缩放图片: {w}x{h} -> {new_w}x{new_h}")
        img = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)

        resized_path = img_path + "_resized.jpg"
        img.save(resized_path, "JPEG", quality=95)
        return resized_path
    except Exception as e:
        logger.warning(f"图片缩放失败: {e}")
        return img_path


def _download_image(image_url, timeout=15):
    """
    下载图片到临时文件，处理webp转换，返回可供OCR使用的本地路径。
    失败返回None。
    """
    try:
        resp = requests.get(image_url, timeout=timeout, stream=True)
        resp.raise_for_status()

        suffix = ".jpg"
        content_type = resp.headers.get("Content-Type", "")
        if "png" in content_type or image_url.endswith(".png"):
            suffix = ".png"
        elif "webp" in content_type or "webp" in image_url:
            suffix = ".webp"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            for chunk in resp.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name

        # webp 转 jpg（PaddleOCR 对 webp 支持不稳定）
        if suffix == ".webp":
            jpg_path = tmp_path + ".jpg"
            Image.open(tmp_path).convert("RGB").save(jpg_path, "JPEG", quality=95)
            os.unlink(tmp_path)
            tmp_path = jpg_path

        return tmp_path

    except Exception as e:
        logger.warning(f"下载失败 {image_url[:60]}...: {e}")
        return None


def _ocr_single_file(file_path):
    """
    对单个本地图片文件执行OCR（线程安全，内部加锁）。
    返回识别到的文字字符串。
    """
    try:
        # 检查是否需要缩放
        ocr_path = _resize_if_needed(file_path)

        with _ocr_lock:
            result = ocr_engine.ocr(ocr_path, cls=True)

        # 清理缩放后的临时文件
        if ocr_path != file_path and os.path.exists(ocr_path):
            os.unlink(ocr_path)

        if not result or not result[0]:
            return ""

        lines = []
        for line_info in result[0]:
            if line_info and len(line_info) >= 2:
                text = line_info[1][0]
                lines.append(text)

        text = "\n".join(lines).strip()
        return text if len(text) >= 2 else ""

    except Exception as e:
        logger.warning(f"OCR识别失败 {file_path}: {e}")
        return ""


def ocr_all_images(image_list):
    """
    对笔记的所有图片执行OCR，严格按图片顺序拼接文字。

    流程：
    1. 并行下载所有图片（I/O密集，适合并行）
    2. 按顺序逐张OCR（PaddleOCR不线程安全，串行保证顺序和稳定性）
    3. 按原始顺序拼接
    """
    if not image_list:
        return ""

    # 第1步：并行下载图片
    local_paths = [None] * len(image_list)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_download_image, url): i
            for i, url in enumerate(image_list)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                local_paths[idx] = future.result()
            except Exception as e:
                logger.warning(f"图片{idx+1}下载异常: {e}")

    # 第2步：按顺序逐张OCR
    texts = []
    for i, path in enumerate(local_paths):
        if path is None:
            logger.debug(f"  图片{i+1}/{len(image_list)} 下载失败，跳过")
            texts.append("")
            continue

        text = _ocr_single_file(path)

        # 清理临时文件
        try:
            os.unlink(path)
        except OSError:
            pass

        if text:
            logger.debug(f"  图片{i+1}/{len(image_list)} OCR成功，{len(text)}字符")
        else:
            logger.debug(f"  图片{i+1}/{len(image_list)} 未识别到文字")

        texts.append(text)

    # 第3步：按顺序拼接（无缝衔接，只过滤空结果）
    return "".join([t for t in texts if t])


def clean_ocr_text(text):
    if not text:
        return ""
    text = text.replace("\n", "")
    return text.strip()


class Data_Spider():

    def __init__(self):
        self.xhs_apis = XHS_Apis()

    def spider_note(self, note_url, cookies_str, proxies=None):
        note_info = None
        try:
            success, msg, note_info = self.xhs_apis.get_note_info(
                note_url, cookies_str, proxies
            )
            if success:
                print(f"DEBUG keys: {note_info.get('data', {}).keys() if isinstance(note_info.get('data'), dict) else type(note_info.get('data'))}")
                note_info = note_info['data']['items'][0]
                note_info['url'] = note_url
                note_info = handle_note_info(note_info)
        except Exception as e:
            success = False
            msg = e
        logger.info(f'爬取笔记信息 {note_url}: {success}, msg: {msg}')
        return success, msg, note_info

    def spider_note_with_ocr(self, note_url, cookies_str, proxies=None):
        success, msg, note_info = self.spider_note(
            note_url, cookies_str, proxies
        )
        if not success or note_info is None:
            return None

        image_list = note_info.get("image_list", [])
        if image_list:
            logger.info(f"正在对 {len(image_list)} 张图片执行OCR...")
            ocr_text = ocr_all_images(image_list)
            ocr_text = clean_ocr_text(ocr_text)
        else:
            ocr_text = ""

        upload_time = note_info.get("upload_time", "")

        result = {
            "title": note_info.get("title", ""),
            "desc": ocr_text,
            "liked_count": str(note_info.get("liked_count", "")),
            "collected_count": str(note_info.get("collected_count", "")),
            "comment_count": str(note_info.get("comment_count", "")),
            "share_count": str(note_info.get("share_count", "")),
            "image_list": image_list,
            "upload_time": upload_time
        }

        return result

    def spider_user_all_note_with_ocr(
        self,
        user_url,
        cookies_str,
        max_notes=20,
        proxies=None
    ):
        note_list = []

        success, msg, all_note_info = self.xhs_apis.get_user_all_notes(
            user_url, cookies_str, proxies
        )

        if not success:
            return [], success, msg

        logger.info(f'用户共 {len(all_note_info)} 条笔记，准备抓取最多 {max_notes} 条')

        note_url_list = []
        for simple_note_info in all_note_info:
            note_url = f"https://www.xiaohongshu.com/explore/{simple_note_info['note_id']}?xsec_token={simple_note_info['xsec_token']}"
            note_url_list.append(note_url)
            if len(note_url_list) >= max_notes:
                break

        # 笔记级并行：多条笔记同时爬取+OCR
        # OCR内部通过锁串行化，不会冲突；HTTP请求仍然并行加速
        results_with_idx = [None] * len(note_url_list)

        with ThreadPoolExecutor(max_workers=NOTE_WORKERS) as executor:
            futures = {
                executor.submit(
                    self.spider_note_with_ocr,
                    url,
                    cookies_str,
                    proxies
                ): i for i, url in enumerate(note_url_list)
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    note = future.result()
                    results_with_idx[idx] = note
                except Exception as e:
                    logger.warning(f"笔记处理异常: {e}")

        # 保持原始顺序
        note_list = [n for n in results_with_idx if n is not None]

        return note_list, True, ""


if __name__ == '__main__':

    load_dotenv()

    cookies_str = os.getenv("COOKIES")

    base_path = init()

    data_spider = Data_Spider()

    user_url = "https://www.xiaohongshu.com/user/profile/689da2b2000000001903ece0?xsec_token=ABm3yCbg0kEdPw_bvDQ4NE50NHPenp-SeMrXTV1lr7r1w%3D&xsec_source=pc_search"

    MAX_NOTES = 700

    proxies = None

    print(f"开始爬取用户最近 {MAX_NOTES} 条笔记")

    notes, success, msg = data_spider.spider_user_all_note_with_ocr(
        user_url,
        cookies_str,
        MAX_NOTES,
        proxies
    )

    if not os.path.exists("datas/json_datas"):
        os.makedirs("datas/json_datas")

    save_path = "datas/json_datas/notes.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=4)

    print(f"完成，共保存 {len(notes)} 条笔记")