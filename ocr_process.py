import os
import json

# 認証ファイルパスを明示的に指定
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "/Users/kinoshitayoshihiro/music_21/sakuhin/_03_ore/composer/marutakesyobou-97c9ac6c8f6f.json"
)

from google.cloud import storage
from google.cloud import vision


def chunk_list(lst, chunk_size):
    """リストを chunk_size ごとのチャンクに分割するジェネレーター"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def ocr_process_image_files(bucket_name, prefix, batch_size=16):
    # Storage クライアントと Vision クライアントの初期化
    storage_client = storage.Client()
    vision_client = vision.ImageAnnotatorClient()

    # 指定したバケットとフォルダ内のオブジェクトを取得
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    requests_list = []
    file_names = []

    # 対象とするファイルの拡張子（png, jpg, jpeg）
    valid_extensions = (".png", ".jpg", ".jpeg")

    # 対象ファイルのみをリストに追加
    for blob in blobs:
        if blob.name.lower().endswith(valid_extensions):
            print(f"Processing: {blob.name}")
            image = vision.Image()
            image.source.image_uri = f"gs://{bucket_name}/{blob.name}"
            request = vision.AnnotateImageRequest(
                image=image,
                features=[vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION)],
            )
            requests_list.append(request)
            file_names.append(blob.name)

    if not requests_list:
        print("対象の画像ファイルが見つかりませんでした。")
        return

    # すべてのリクエストをバッチに分割して処理する
    all_results = []
    for batch_index, batch_requests in enumerate(
        chunk_list(requests_list, batch_size), start=1
    ):
        print(f"Processing batch {batch_index} ( {len(batch_requests)} images )...")
        response = vision_client.batch_annotate_images(requests=batch_requests)
        all_results.extend(response.responses)

    # 各ファイルの OCR 結果を表示
    # ここでは、requests_list と file_names の順序が同じである前提です。
    for file_name, res in zip(file_names, all_results):
        print(f"File: {file_name}")
        if res.error.message:
            print("Error:", res.error.message)
        else:
            if res.text_annotations:
                print("Detected text:")
                print(res.text_annotations[0].description)
            else:
                print("No text detected.")
        print("-" * 40)


if __name__ == "__main__":
    bucket_name = "bungo-syousetu"
    prefix = "裏の木戸はあいている/"  # 例: "images/pngs/"
    ocr_process_image_files(bucket_name, prefix)
