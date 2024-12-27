#!/usr/bin/env python
# -*- coding: utf-8 -*-

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image, ImageChops
import os
import sys
import io
import argparse
import shutil


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def pdf_to_images(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.pil_tobytes(format='png')
        img = Image.open(io.BytesIO(img_data))
        img.save(os.path.join(output_dir, f"page_{str(page_num + 1).zfill(3)}.png"))


def mark_differences(img1_path, img2_path, output_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 画像をグレースケールに変換
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 画像の差分を計算
    diff = cv2.absdiff(gray1, gray2)

    # 差分を2値化
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 差分を赤でマーキング
    ecout = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        ecout += 1

    # マーキングされた画像を保存
    if ecout > 0:
        cv2.imwrite(output_path, img1)

    return ecout


def get_param():
    parser = argparse.ArgumentParser(description="compare pdf")
    parser.add_argument("pdf1", help="input PDF1")
    parser.add_argument("pdf2", help="input PDF2")
    parser.add_argument("-o", "--output", help="output folder")
    args = parser.parse_args()

    return args


def main():
    args = get_param()

    pdf1_path = args.pdf1
    pdf2_path = args.pdf2

    output_dir1 = args.output + "/images_pdf1"
    output_dir2 = args.output + "/images_pdf2"
    diff_output_dir = args.output + "/diff_pdf"

    # 出力フォルダをクリア
    clear_folder(output_dir1)
    clear_folder(output_dir2)
    clear_folder(diff_output_dir)

    pdf_to_images(pdf1_path, output_dir1)
    pdf_to_images(pdf2_path, output_dir2)

    images_pdf1 = sorted(os.listdir(output_dir1))
    images_pdf2 = sorted(os.listdir(output_dir2))

    # Check if the number of images in both directories is the same
    if len(images_pdf1) != len(images_pdf2):
        print("Error: The number of pages in the PDFs does not match.")
        sys.exit(1)

    total_pages = len(images_pdf1)
    print(f"元のページ数: {total_pages}")
    changed_pages = []

    for page_num, (img1, img2) in enumerate(zip(images_pdf1, images_pdf2), start=1):
        img1_path = os.path.join(output_dir1, img1)
        img2_path = os.path.join(output_dir2, img2)
        output_path = os.path.join(diff_output_dir, f"diff_{img1}")
        if mark_differences(img1_path, img2_path, output_path) > 0:
            changed_pages.append(page_num)
            print(f"差分がマーキングされた画像が{output_path}として保存されました。")

    if changed_pages:
        print(f"変更のあったページ: {', '.join(map(str, changed_pages))}")
    else:
        print("変更のあったページはありませんでした。")


if __name__ == "__main__":
    main()
