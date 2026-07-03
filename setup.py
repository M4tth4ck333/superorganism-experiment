import sys
import os
from cx_Freeze import setup, Executable

base = "Win32GUI" if sys.platform == "win32" else None

submodule_path = os.path.abspath("torrent_health_and_investment")
sys.path.append(submodule_path)

include_files = [
    ("logging_config.json", "logging_config.json"),
    ("ui/", "ui/"),
    ("libsodium.dll", "libsodium.dll"),
    ("crowdsourced_learn_to_rank/ltr-benchmarking/", "ltr-benchmarking/"),
]

packages = [
    "aiohttp",
    "bencodepy",
    "cryptography",
    "httpx",
    "libnacl",
    "libtorrent",
    "ipv8",
    "PySide6",
    "matplotlib",
    "numpy",
    "sklearn",
    "lightgbm",
    "xgboost"
]

python_excludes = [
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtWebView",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DExtras",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
]

dll_excludes = [
    "Qt6Multimedia.dll",
    "Qt6MultimediaWidgets.dll",
    "Qt6WebView.dll",
    "Qt6WebEngineCore.dll",
    "Qt6WebEngineWidgets.dll",
    "Qt63DCore.dll",
    "Qt63DRender.dll",
    "Qt63DInput.dll",
    "Qt63DLogic.dll",
    "Qt63DExtras.dll",
    "Qt6Qml.dll",
    "Qt6QmlModels.dll",
    "Qt6Quick.dll",
    "Qt6QuickWidgets.dll",
    "Qt6Pdf.dll",
    "Qt6Designer.dll",
    "Qt6VirtualKeyboard.dll",
]

build_exe_options = {
    "include_files": include_files,
    "packages": packages,
    "include_path": [submodule_path],
    "zip_includes": [
        ("torrent_health_and_investment/healthchecker/", "healthchecker/"),
        ("crowdsourced_learn_to_rank/ltr-benchmarking/", "crowdsourced_learn_to_rank/ltr-benchmarking/"),
        ("crowdsourced_learn_to_rank/", "crowdsourced_learn_to_rank/"),
    ],
    "excludes": python_excludes,
    "bin_excludes": dll_excludes,
}

setup(
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "main.py",
            base=base,
            target_name="SuperorganismExperiment.exe"
        )
    ],
)
