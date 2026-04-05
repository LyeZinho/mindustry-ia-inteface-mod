from setuptools import setup, find_packages

setup(
    name="mindustry-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.7.0",
        "pyautogui>=0.9.53",
        "pillow>=9.5.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
