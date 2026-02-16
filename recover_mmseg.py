import os
import sys
import subprocess
import re
from pathlib import Path

def recover_mmseg_init():
    # 1. Find the file location
    mmseg_path = ""
    try:
        result = subprocess.check_output([sys.executable, "-m", "pip", "show", "mmsegmentation"], stderr=subprocess.DEVNULL).decode()
        for line in result.split('\n'):
            if line.startswith('Location: '):
                mmseg_path = os.path.join(line.split(': ')[1].strip(), "mmseg/__init__.py")
                break
    except Exception as e:
        print(f"❌ Could not find mmsegmentation: {e}")
        return

    if not mmseg_path or not os.path.exists(mmseg_path):
        print("❌ mmseg/__init__.py not found.")
        return

    print(f"📍 Found mmsegmentation init at: {mmseg_path}")
    
    # 2. Read CURRENT content
    with open(mmseg_path, 'r') as f:
        content = f.read()

    print(f"📄 file size before: {len(content)} bytes")

    # 3. Aggressive Cleaning Strategy
    # We want to remove *everything* from the first `mmcv_min_version =` down to the `__all__ =` 
    # and replace it with our clean block.
    
    # Pattern to match the entire messy block of version checks and assertions
    pattern = r"(mmcv_min_version\s*=.*?)(?=__all__\s*=)"
    
    clean_block = """
mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmcv_version = digit_version(mmcv.__version__)

mmengine_min_version = digit_version(MMENGINE_MIN)
mmengine_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmengine_version = digit_version(mmengine.__version__)

# CLEANED BY DEEPROOF RECOVERY SCRIPT
\n"""

    new_content = re.sub(pattern, clean_block, content, flags=re.DOTALL)
    
    # If regex failed (maybe the file is too broken), we fallback to FULL OVERWRITE
    if len(new_content) == len(content) or "IndentationError" in new_content: 
        print("⚠️ Regex didn't catch the block. Falling back to FULL OVERWRITE.")
        new_content = """# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__, version_info

MMCV_MIN = '2.0.0rc4'
MMCV_MAX = '2.2.0'
MMENGINE_MIN = '0.7.1'
MMENGINE_MAX = '1.0.0'

mmcv_min_version = digit_version(MMCV_MIN)
mmcv_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmcv_version = digit_version(mmcv.__version__)

mmengine_min_version = digit_version(MMENGINE_MIN)
mmengine_max_version = digit_version('9.9.9') # OVERRIDE by DeepRoof
mmengine_version = digit_version(mmengine.__version__)

__all__ = ['__version__', 'version_info', 'digit_version']
"""

    # 4. Overwrite
    try:
        with open(mmseg_path, 'w') as f:
            f.write(new_content)
        print("✅ SUCCESSFULLY OVERWROTE mmseg/__init__.py")
        print("⚠️ PLEASE RESTART YOUR KERNEL NOW.")
    except Exception as e:
        print(f"❌ Failed to write file: {e}")

if __name__ == "__main__":
    recover_mmseg_init()
