from setuptools import setup, find_packages

setup(
    name='illust_salmap',  # パッケージ名
    version='0.1.0',  # パッケージのバージョン
    description='A short description of your project',  # プロジェクトの簡単な説明
    long_description=open('README.md').read(),  # 長い説明（通常は README.md から取得）
    long_description_content_type='text/markdown',  # 長い説明のコンテンツタイプ（Markdownの場合）
    author='Your Name',  # 作成者名
    author_email='your.email@example.com',  # 作成者のメールアドレス
    url='https://github.com/your_username/your_project',  # プロジェクトのURL（GitHubなど）
    packages=find_packages(),  # パッケージを自動的に検索
    install_requires=[  # 必要な依存ライブラリ
        "torchsummary",
        "torch",
        "torchvision",
        "scikit-learn",
        "gdown",
    ],
    classifiers=[  # PyPIにおける分類情報
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # 必要なPythonバージョン
)
