import os, gradio as gr, numpy as np
from pathlib import Path
import sys

os.environ.setdefault("DISABLE_MPS","1")
os.environ.setdefault("SBER_BATCH","32")
os.environ.setdefault("DET_CHUNK","256")

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import detector.detector as D
from detector.detector import JailbreakDetector
from detector.helpers import is_single_turn_text

det = JailbreakDetector()

thr_single = gr.Slider(0.0, 1.0, value=det.threshold_single, step=0.01, label="Single-turn threshold")
thr_multi  = gr.Slider(0.0, 1.0, value=det.threshold_multi,  step=0.01, label="Multi-turn threshold")
show_ch    = gr.Checkbox(value=True, label="Show channel scores (BOW/EMB/FEAT)")

def score(txt:str, thr_s:float, thr_m:float, show_channels:bool):
    det.threshold_single = float(thr_s)
    det.threshold_multi  = float(thr_m)
    if not txt.strip():
        return {"prob":0,"pred":0,"turn":"-","thr":0, "channels":{}}

    # Ensemble olasılığı
    prob = float(det.predict_proba([txt])[0])
    single = is_single_turn_text(txt)
    thr = det.threshold_single if single else det.threshold_multi
    pred = int(prob >= thr)
    out = {"prob":prob, "pred":pred, "turn":"single" if single else "multi", "thr":thr}

    # Kanal skorları (modül-level fonksiyonlar üzerinden)
    if show_channels:
        p_bow = float(D._bow_lr_prob([txt])[0])
        pr_txt, hi_txt, E_p, E_h = D._prepare_prompt_and_history([txt], det.window)
        out_scores = D._outlier_scores(E_p)
        p_emb  = float(D._sbert_emb_prob(E_p)[0])
        p_feat = float(D._sbert_feat_prob([txt], E_p, E_h, pr_txt, hi_txt, out_scores)[0])
        out["channels"] = {"bow":p_bow, "emb":p_emb, "feat":p_feat}
    return out

demo = gr.Interface(
    fn=score,
    inputs=[gr.Textbox(lines=12, label="Prompt / multi-turn"),
            thr_single, thr_multi, show_ch],
    outputs=gr.JSON(label="Result"),
    title="SLM Jailbreak Detector"
)

# demo.launch(server_name="127.0.0.1", server_port=0, show_error=True)
demo.launch(share=True)