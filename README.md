## Quickstart

Follow the steps below to set up and execute the DSPM pipeline.

### 1. Clone the Repository
```bash
git clone https://github.com/zatchbell1311-wq/dynamic-semantic-patch-memory.git
cd dynamic-semantic-patch-memory
2. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
3. Configure API Key

Set your Groq API key as an environment variable:

export GROQ_API_KEY="your_groq_api_key_here"
4. Run the Pipeline
python run_pipeline.py
Output Files

After execution, the following artifacts will be generated:

File	Description
dspm_results.csv	Complete evaluation results and metrics
fig_main_results.png	Combined visualization: token usage, TRR, per-turn growth, and consistency
fig_ablation.png	Ablation study across DSPM components
fig_baselines.png	Comparison between baseline methods (Raw, Truncation, Compression, DSPM)
fig_patch_types.png	Distribution of token allocation across patch types
fig_error_robustness.png	Error handling and robustness statistics
error_log.json	Logged API errors for analysis
Citation

If you use this work, please cite:

@article{dubey2026dspm,
  title   = {DSPM: Dynamic Semantic Patch Memory for Token-Efficient Long-Context Reasoning in Large Language Models},
  author  = {Dubey, Dhruv and Pandey, Adarsh and Vishwakarma, Anurag and Srivastava, Kartikay},
  journal = {IEEE International Conference on Systems, Man, and Cybernetics (SMC 2026)},
  year    = {2026},
  note    = {Under Review}
}
Authors

Dhruv Dubey
Bansal Institute of Engineering and Technology, AKTU, Lucknow, India
ORCID: https://orcid.org/0009-0004-5510-9000

Email: dhruvdubey1311@gmail.com

License

This project is licensed under the MIT License.
See the LICENSE file for more details.



If you want, I can next convert this into a full top-tier README with abstract, architecture diagram section, badges, and usage examples.
