# --- START OF FILE utilities/safe_get.py ---
from typing import Any, Callable, Optional, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")  # ジェネリック型のための型変数

# SENTINELオブジェクトは、Noneが有効なデフォルト値である場合と区別するために使用できますが、
# 今回のsafe_getでは、default引数で直接デフォルト値を指定する形なので、必須ではありません。
# より複雑なケース（例えば、キーが存在しない場合と、キーは存在するが値がNoneの場合を区別したいなど）
# で役立ちます。ここではシンプルに保ちます。


def safe_get(
    data: dict,
    key_path: str,
    *,
    default: T = None,  # デフォルト値の型をジェネリックに
    cast_to: Optional[Callable[[Any], T]] = None,  # キャスト関数の型もジェネリックに
    log_name: str = "safe_get",  # ログ出力時の識別子
) -> T:  # 戻り値の型もジェネリックに
    """
    辞書からネストしたキーパスを使って安全に値を取得します。
    キーが存在しない場合、値がNoneの場合、または型キャストに失敗した場合は、
    指定されたデフォルト値を返します。

    Args:
        data: 対象の辞書。
        key_path: ドット区切りのキーパス (例: "a.b.c")。
        default: 値が見つからない場合や処理に失敗した場合に返すデフォルト値。
        cast_to: 取得した値を指定された型に変換する関数 (例: float, int, str)。
                 Noneの場合は型変換を行いません。
        log_name: ログメッセージに含める識別名。

    Returns:
        取得・変換された値、またはデフォルト値。
    """
    current_value: Any = data
    keys = key_path.split(".")

    for i, key in enumerate(keys):
        if isinstance(current_value, dict) and key in current_value:
            current_value = current_value[key]
        else:
            # キーパスの途中でキーが見つからなかった場合
            logger.debug(
                f"{log_name}: Key '{key}' not found in path '{'.'.join(keys[:i+1])}' from '{key_path}'. Returning default: {default!r}"
            )
            return default

    # キーパス全体が見つかった後、値がNoneの場合のチェック
    if current_value is None:
        logger.debug(
            f"{log_name}: Value for key_path '{key_path}' is None. Returning default: {default!r}"
        )
        return default

    # 型キャスト処理
    if cast_to:
        try:
            casted_value = cast_to(current_value)
            return casted_value
        except (ValueError, TypeError) as e_cast:
            logger.warning(
                f"{log_name}: Failed to cast value {current_value!r} for key_path '{key_path}' to {cast_to.__name__}. Error: {e_cast}. Returning default: {default!r}"
            )
            return default
        except Exception as e_unexpected_cast:  # その他の予期せぬキャストエラー
            logger.error(
                f"{log_name}: Unexpected error casting value {current_value!r} for key_path '{key_path}' to {cast_to.__name__}. Error: {e_unexpected_cast}. Returning default: {default!r}",
                exc_info=True,
            )
            return default

    return current_value  # キャストなし、またはキャスト成功


# --- END OF FILE utilities/safe_get.py ---
