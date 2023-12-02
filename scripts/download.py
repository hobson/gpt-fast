# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from requests.exceptions import HTTPError
import sys
from pathlib import Path
from typing import Optional

def hf_download(repo_id: Optional[str] = None, hf_token: Optional[str] = None) -> None:
    from huggingface_hub import snapshot_download
    os.makedirs(f"checkpoints/{repo_id}", exist_ok=True)
    try:
        snapshot_download(repo_id, repo_type='model', token=hf_token)
    except HTTPError as e:
        if e.response.status_code == 401:
            print(f"Invalid --hf_token={hf_token} or --repo_id={repo_id}")
            print("You need to pass a valid `--hf_token=hf_...` and `--repo_id=meta-llama/..` to download private model checkpoints.")
        else:
            raise e

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Download data from HuggingFace Hub.')
    parser.add_argument('--repo_id', type=str, default="meta-llama/Llama-2-7b-chat-hf", help='Repository ID for model to download.')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace API token.')

    args = parser.parse_args()
    hf_download(args.repo_id, args.hf_token)
