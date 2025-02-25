import filecmp
import shutil

import huggingface_hub as hf_hub

draft_model_id = "OpenVINO/Llama-3.1-8B-Instruct-FastDraft-150M-int8-ov"
draft_model_path = Path("Llama_3.1_FastDraft")

if not draft_model_path.exists():
    hf_hub.snapshot_download(draft_model_id, local_dir=draft_model_path)

# We need tokenizers to match between the target and draft model so we apply this workaround
if not filecmp.cmp(str(model_dir / "openvino_tokenizer.xml"), str(draft_model_path / "openvino_tokenizer.xml"), shallow=False):
    for fname in ["openvino_tokenizer.xml", "openvino_tokenizer.bin", "openvino_detokenizer.xml", "openvino_detokenizer.bin"]:
        shutil.copy(model_dir / fname, draft_model_path / fname)