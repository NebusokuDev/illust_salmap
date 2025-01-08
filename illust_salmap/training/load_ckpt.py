from importlib import import_module
from pathlib import Path
import torch

from illust_salmap.training.saliency_model import SaliencyModel


def instantiate_from(
        model_name: str,
        module_name: str,
        ckpt_path: str | Path,
        on_instantiate: callable = None
        ) -> SaliencyModel:
    """
    指定されたモデルクラスとモジュールから、サリエンシーモデルのチェックポイントをロードします。

    Args:
        on_instantiate: モデル生成時のフック。デフォルトはそのままモデルを生成。
        model_name (str): モジュール内のモデルクラス名。
        module_name (str): モジュール名。
        ckpt_path (str | Path): チェックポイントのパス。

    Returns:
        SaliencyModel: ロード済みのサリエンシーモデル。
    """
    # デフォルトのon_instantiate関数を設定
    if on_instantiate is None:
        on_instantiate = lambda model: model  # デフォルトでは何も変更せずに返す

    module_name = f"illust_salmap.models.{module_name}"
    try:
        # モジュールのインポート
        module = import_module(module_name)
        model_cls = getattr(module, model_name)  # モジュール内のモデルクラスを取得
        model = on_instantiate(model_cls())  # モデルインスタンスを生成
        # サリエンシーモデルのインスタンスをロード
        saliency_model = SaliencyModel(model).load_from_checkpoint(str(ckpt_path))
        return saliency_model
    except ModuleNotFoundError as e:
        raise ImportError(f"Failed to import module: {module_name}") from e
    except AttributeError as e:
        raise ImportError(f"Class {model_name} not found in module: {module_name}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}: {e}") from e


def get_best_ckpt_path(root: str | Path, key: str, sort: str = "asc") -> Path:
    """
    指定されたディレクトリから、特定のキーでソートした最良のチェックポイントパスを取得します。

    Args:
        root (str | Path): チェックポイントが保存されているディレクトリのパス。
        key (str): ソート基準となるチェックポイントのキー。
        sort (str): ソート順。'asc' (昇順) または 'desc' (降順)。

    Returns:
        Path: 最良のチェックポイントファイルのパス。
    """
    root = Path(root)

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    ckpt_files = list(root.glob("*.ckpt"))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in: {root}")

    # チェックポイントとその対応するファイルパスをロード
    checkpoints = []
    for file in ckpt_files:
        try:
            ckpt = torch.load(file, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file: {file}. Error: {e}")

        if key not in ckpt:
            raise KeyError(f"Key '{key}' not found in checkpoint: {file}")

        checkpoints.append((ckpt, file))

    # 指定されたキーでソート
    reverse = (sort == "desc")
    sorted_checkpoints = sorted(checkpoints, key=lambda item: item[0][key], reverse=reverse)

    # 最良のチェックポイントのファイルパスを返す
    return sorted_checkpoints[0][1]


if __name__ == '__main__':
    # チェックポイントパスを取得
    ckpt_path = get_best_ckpt_path("./ckpt/unet", "loss", sort="asc")
    # モデルをインスタンス化
    model = instantiate_from("UNet", "unet_v2", ckpt_path)
    print(model)
