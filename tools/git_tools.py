# git_tools.py
import subprocess
import logging

logger = logging.getLogger(__name__)


def get_git_diff_staged():
    """Gitのステージングエリアにある変更の差分を取得します。"""
    try:
        # git diff --staged コマンドを実行
        result = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True,
            check=True,  # エラーコードが0以外なら例外を発生
        )
        logger.info("Successfully retrieved staged git diff.")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error getting git diff: {e.stderr}")
        if "not a git repository" in e.stderr.lower():
            return "Error: Not a git repository or no git configured."
        elif (
            not e.stdout and not e.stderr
        ):  # ステージングエリアに何もない場合もエラーになることがある
            logger.info("No changes staged for diff.")
            return "No changes staged."
        return f"Error getting git diff: {e.stderr}"
    except FileNotFoundError:
        logger.error(
            "Git command not found. Make sure Git is installed and in your PATH."
        )
        return "Error: Git command not found."
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in get_git_diff_staged: {e}", exc_info=True
        )
        return f"Unexpected error: {e}"


if __name__ == "__main__":
    # テスト用
    print("--- Staged Diff ---")
    diff_output = get_git_diff_staged()
    print(diff_output)
    if "Error" in diff_output:
        print(
            "\nNote: To test effectively, stage some changes in a git repository first."
        )
        print("Example: echo 'test' > test_file.txt && git add test_file.txt")
# git_tools.py (続き)


def commit_changes_with_message(message: str):
    """指定されたコミットメッセージで変更をコミットします。"""
    if not message or not message.strip():
        logger.error("Commit message cannot be empty.")
        return "Error: Commit message cannot be empty."
    try:
        # git commit -m "message" コマンドを実行
        result = subprocess.run(
            ["git", "commit", "-m", message], capture_output=True, text=True, check=True
        )
        logger.info(f"Successfully committed changes with message: {message}")
        # コミット成功時は、コミットハッシュなどの情報を返すとより親切
        # 例: result.stdout (git commit の出力は通常ブランチ情報など)
        #     git log -1 --pretty=%H で最新のコミットハッシュを取得できる
        commit_hash_result = subprocess.run(
            ["git", "log", "-1", "--pretty=%H"],
            capture_output=True,
            text=True,
            check=True,
        )
        return (
            f"Successfully committed. Commit hash: {commit_hash_result.stdout.strip()}"
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Error committing changes: {e.stderr}")
        if "nothing to commit" in e.stderr.lower():
            return "No changes staged to commit."
        return f"Error committing changes: {e.stderr}"
    except FileNotFoundError:
        logger.error("Git command not found.")
        return "Error: Git command not found."
    except Exception as e:
        logger.error(
            f"An unexpected error occurred in commit_changes_with_message: {e}",
            exc_info=True,
        )
        return f"Unexpected error: {e}"


if __name__ == "__main__":
    # テスト用 (get_git_diff_staged のテストの後に実行すると良い)
    print("\n--- Committing Changes (Test) ---")
    # テストの際は、実際にコミットされてしまうので注意
    # staged_diff = get_git_diff_staged()
    # if "No changes staged" not in staged_diff and "Error" not in staged_diff:
    #     commit_message = "Test commit via function call"
    #     commit_result = commit_changes_with_message(commit_message)
    #     print(commit_result)
    # else:
    #     print("Skipping commit test as no changes are staged or diff error occurred.")
    pass  # コミットテストは手動で確認
