---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/zatchbell1311-wq/dynamic-semantic-patch-memory.git
cd dynamic-semantic-patch-memory
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Set your Groq API key
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the full evaluation
```bash
python run_pipeline.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `dspm_results.csv` | Full numeric results table |
| `fig_main_results.pdf` | 2×2 panel: token usage, TRR, per-turn growth, consistency |
| `fig_ablation.pdf` | Per-technique ablation bar chart |
| `fig_baselines.pdf` | Method comparison (Raw / Truncation / Compression / DSPM) |
| `fig_patch_types.pdf` | Budget allocation pie chart |
| `fig_error_robustness.pdf` | Error & robustness stats per dialogue |
| `error_log.json` | API error log for robustness analysis |

---

## Citation

If you use this work, please cite:
```bibtex
@article{dubey2026dspm,
  title     = {DSPM: Dynamic Semantic Patch Memory for Token-Efficient
               Long-Context Reasoning in Large Language Models},
  author    = {Dubey, Dhruv and Pandey, Adarsh and
               Vishwakarma, Anurag and Srivastava, Kartikay},
  journal   = {IEEE International Conference on Systems, Man, and Cybernetics
               (SMC 2026) — Under Review},
  year      = {2026},
}
```

---


## Authors

**Dhruv Dubey** — Bansal Institute of Engineering & Technology, AKTU, Lucknow, India  
ORCID: [0009-0004-5510-9000](https://orcid.org/0009-0004-5510-9000)  
Email: dhruvdubey1311@gmail.com

**Collaborators:** Adarsh Pandey · Anurag Vishwakarma · Kartikay Srivastava

---

## License

MIT License — see [LICENSE](LICENSE) for details.
