"""
dialogues.py — Hardcoded evaluation dialogues + dynamic dialogue generator.

7 hardcoded technical dialogues (8 turns each):
  1. REST_API_Design
  2. ML_Model_Training
  3. IoT_Pipeline
  4. Cybersecurity_IR
  5. Supply_Chain_Opt
  6. Medical_Workflow
  7. Project_Mgmt_Jira

Plus a dynamic generator for additional Groq-generated dialogues.
"""

import re
import json
from typing import List, Tuple, Dict

from dspm.evaluator import generate_llm_response
from dspm.extractor import GroqClientPool

# ── 7 Hardcoded Dialogues ─────────────────────────────────────────────────

HARDCODED_DIALOGUES: Dict[str, List[Tuple[str, str]]] = {

"REST_API_Design": [
    ("user", "We need to build a REST API for an e-commerce backend. Core entities are products, orders, users. Use FastAPI with PostgreSQL and Redis for caching. JWT auth required. Target latency under 100ms for reads."),
    ("asst", "FastAPI chosen; PostgreSQL with async SQLAlchemy ORM; Redis cache with 60s TTL for product listings. JWT with RS256 signing. Products GET /products/{id} will target ≤80ms with Redis hit."),
    ("user", "Add AWS S3 for product image storage. Product schema needs: id, name, price (Decimal), stock_count, image_url, category_id. Image upload must be pre-signed URL, max 5MB."),
    ("asst", "S3 pre-signed URLs with 15min expiry; boto3 integration added. Product schema defined with Decimal(10,2) for price. Image validation: MIME-type check + size gate at API layer before S3 upload."),
    ("user", "Orders must support partial fulfilment. Order status: PENDING, CONFIRMED, PARTIALLY_FULFILLED, FULFILLED, CANCELLED. Payment via Stripe webhooks. Inventory lock must be optimistic, not pessimistic."),
    ("asst", "Order FSM with five states implemented. Stripe webhook endpoint POST /webhooks/stripe with signature verification. Optimistic locking via version column on inventory table; conflict returns HTTP 409."),
    ("user", "Deployment on ECS Fargate. CI/CD via GitHub Actions. Containerise with Docker. DB migrations via Alembic. Rate limit: 1000 req/min per user. Final architecture sign-off needed."),
    ("asst", "Docker multi-stage build; ECS Fargate task def with 0.5 vCPU / 1GB RAM. GitHub Actions: test → lint → build → push ECR → deploy. Alembic migration step in CI. Rate limiting via Redis sliding window at API gateway layer."),
],

"ML_Model_Training": [
    ("user", "Build a customer support ticket classifier. Binary: urgent vs non-urgent. Dataset is 50K tickets, heavily imbalanced — 80% non-urgent. Use BERT fine-tuning. F1 target ≥ 0.88 on minority class."),
    ("asst", "BERT-base-uncased chosen. Class weights: urgent=4.0, non-urgent=1.0 to handle 80/20 split. AdamW with lr=2e-5, warmup_steps=500. Max sequence length 256 tokens. Batch size 16 on T4."),
    ("user", "Add focal loss to handle imbalance better. Also need ONNX export for production inference. Latency constraint: ≤30ms per prediction on CPU at inference time."),
    ("asst", "Focal loss with gamma=2.0 added alongside class weights. ONNX export via torch.onnx.export with opset 14. ONNX Runtime with INT8 quantisation targets ≤25ms CPU latency. Validated on Intel Xeon."),
    ("user", "Evaluation must include per-class precision/recall/F1 plus confusion matrix. Add MLflow for experiment tracking. Model versioning in S3. Minimum 5-fold cross-validation required."),
    ("asst", "MLflow tracking server configured; per-experiment metrics: precision, recall, F1, AUC-ROC logged per fold. StratifiedKFold(5) with class stratification. Best model checkpoint to S3 via mlflow.log_artifact."),
    ("user", "Production needs canary deployment — route 10% traffic to new model, rest to stable. Rollback trigger: if F1 drops >5% vs baseline in 1hr window. Monitoring via CloudWatch custom metrics."),
    ("asst", "Canary split via weighted ALB target groups (10/90). Lambda function monitors CloudWatch custom metric ModelF1Score every 15min; auto-rollback if delta >0.05. Baseline F1 stored in Parameter Store."),
],

"IoT_Pipeline": [
    ("user", "Real-time sensor pipeline: 500 IoT devices, each emitting events at 10Hz. Events: temperature, pressure, vibration. Apache Kafka as message broker. Must handle 5000 events/sec sustained load."),
    ("asst", "Kafka cluster: 3 brokers, topic partitions=15 (3x device count heuristic). Producer batch.size=16384, linger.ms=5 for throughput. Consumer group with 5 Flink workers scales to 5K events/sec with headroom."),
    ("user", "Serialisation must be Avro with Schema Registry. Flink stream processing: sliding window of 60s, step 10s for anomaly detection. Alert if temperature > 85°C or vibration RMS > 0.5g. Store raw events in ClickHouse."),
    ("asst", "Avro schemas registered in Confluent Schema Registry; backward compatibility enforced. Flink SlidingEventTimeWindows(60s,10s) with watermark tolerance 2s. ClickHouse MergeTree table partitioned by device_id, toYYYYMM(event_time)."),
    ("user", "Backpressure handling is critical — if Flink lag > 10K messages, throttle producers. Dead letter queue for malformed events. ClickHouse retention: hot tier 30 days, cold tier S3 via TTL rule."),
    ("asst", "Flink checkpointing every 30s with RocksDB state backend for fault tolerance. Consumer lag monitored via Kafka consumer group lag metric; Kafka Quotas API throttles producers when lag >10K. DLQ topic with 7-day retention. ClickHouse TTL: hot 30d, S3 via MOVE TO DISK tiering."),
    ("user", "Add ML inference for predictive maintenance. Model: LightGBM trained on 90 days historical. Inference via Flink async I/O. SLA: inference latency ≤50ms p99. Deploy model as gRPC microservice."),
    ("asst", "LightGBM model served via Triton Inference Server over gRPC. Flink AsyncFunction with 100ms timeout; fallback to rule-based alert if inference unavailable. Triton model warmup prevents cold-start latency. p99 latency target validated via Locust load test."),
],

"Cybersecurity_IR": [
    ("user", "Design an incident response pipeline for a mid-size fintech. Assume SOC team of 5 analysts. SIEM is Splunk. Threat intelligence via MISP. Need automated triage for alerts; target: 80% of P3/P4 alerts auto-resolved."),
    ("asst", "Splunk ES with adaptive response actions. MISP integration via ThreatIntelligence add-on. Playbook automation via Splunk SOAR (Phantom). Decision tree: hash reputation → IP geo-block check → user risk score → auto-resolve if all low."),
    ("user", "SOAR playbooks must integrate PagerDuty for P1/P2 escalation. Containment action: auto-isolate endpoint via CrowdStrike Falcon API if malware confirmed. Evidence retention: 90 days minimum for compliance (PCI-DSS)."),
    ("asst", "SOAR→PagerDuty webhook on severity P1/P2 trigger. CrowdStrike RTR API containment playbook with dual-approval for production hosts. Splunk index retention set to 90 days; cold storage to S3 Glacier at 91 days for 7yr PCI-DSS requirement."),
    ("user", "Threat hunting must run weekly automated hunts using Sigma rules. Map all detections to MITRE ATT&CK. False positive rate target <5%. Add metrics dashboard for MTTR, MTTD, alert volume."),
    ("asst", "SigmaHQ rules converted to Splunk SPL via sigma-cli. ATT&CK Navigator layer auto-updated on each detection hit. FP suppression via tuning workbench targeting <5% FPR baseline. Splunk dashboard: MTTR, MTTD, alert-to-ticket ratio, coverage heatmap."),
    ("user", "Red team exercise needed quarterly. Purple team collaboration: share IOCs between red/blue in real time. Document all runbooks in Confluence. Final gap: insider threat detection using UEBA."),
    ("asst", "Splunk UBA deployed for insider threat; baseline 30-day behaviour model per user. Anomaly score >7.5 triggers SOAR playbook. Red team findings auto-ingested as MISP events. Confluence runbook template with required fields: detection, containment, eradication, recovery, lessons-learned."),
],

"Supply_Chain_Opt": [
    ("user", "We run a 3-tier supply chain: suppliers → warehouses → retail. 200 SKUs, 15 warehouses, 500 retail nodes. Need demand forecasting + inventory optimisation. Current stockout rate is 12%, target <3%."),
    ("asst", "Demand forecasting: Temporal Fusion Transformer (TFT) with 90-day lookback, 30-day horizon. Features: historical sales, promotions, seasonality index, external weather. Inventory policy: continuous review (s,S) per SKU-warehouse pair."),
    ("user", "Add supply uncertainty modelling — lead times vary ±40%. Safety stock must cover 95th percentile demand during lead time. Replenishment triggers must integrate with SAP via REST API. Batch processing daily at 02:00 UTC."),
    ("asst", "Lead time distribution: empirical CDF per supplier, sampled via Monte Carlo (10K iterations) for safety stock calculation. z-score=1.645 for 95% service level. SAP integration via SAP Business Technology Platform REST API; OAuth 2.0 client credentials flow."),
    ("user", "Cost objective: minimise total landed cost = purchasing + holding (18% annually) + stockout penalty ($50/unit/day). Solve via linear programming with PuLP. Constraint: warehouse capacity ≤ 85% utilisation."),
    ("asst", "PuLP LP model: objective minimises purchasing+holding+penalty. Constraints: capacity ≤85%, demand coverage ≥95th percentile, minimum order quantities per supplier contract. Solver: CBC with timeout 120s. Warm-start from previous day solution."),
    ("user", "Dashboard needed: supply chain health score, per-warehouse fill rate, supplier on-time delivery %, projected stockout risk by SKU. Export to PowerBI. Alert if any warehouse fill rate drops below 90%."),
    ("asst", "Health score composite: 0.4×fill_rate + 0.3×OTD + 0.2×turnover + 0.1×forecast_accuracy. PowerBI dataset via REST push API, refreshed post-daily run. PagerDuty alert webhook if fill_rate <0.90 at any warehouse node."),
],

"Medical_Workflow": [
    ("user", "Build a clinical decision support system for sepsis early warning. Input: EHR data — vitals, labs, medications. Target: alert 6 hours before sepsis onset. Must meet HIPAA. Hospital has Epic EHR."),
    ("asst", "Model: XGBoost trained on MIMIC-IV dataset, features: qSOFA score components, lactate trend, WBC, CRP, news2 score. Epic FHIR R4 API integration. HIPAA: all data processing in AWS GovCloud, encryption at rest (AES-256) and in transit (TLS 1.3)."),
    ("user", "Alert fatigue is critical — current sepsis systems have 70% false positive rate. Target FPR <20% at 85% sensitivity. Add explainability via SHAP values for each alert. Nursing staff must see top-3 contributing factors."),
    ("asst", "Threshold tuned on validation set for 85% sensitivity / <20% FPR operating point. SHAP TreeExplainer computes feature contributions per prediction; top-3 displayed in Epic SmartPhrase. Alert suppressed if identical alert fired <4hrs prior for same patient."),
    ("user", "Model must be retrained monthly on new hospital data. Minimum 500 new sepsis cases before retraining trigger. Data drift monitoring: alert if PSI >0.2 on any top-5 feature. Model governance: FDA SaMD class II pathway."),
    ("asst", "MLflow registry with staging/production/archived stages. Retraining trigger: 500 new labelled cases OR PSI >0.2 on monitored features. DVC for data versioning. FDA SaMD pathway: QMS documentation, 510(k) substantial equivalence strategy, IEC 62304 lifecycle compliance."),
    ("user", "Prospective validation trial needed: 90-day shadow mode in 2 ICUs, compare alerts vs clinician decisions. Primary endpoint: time-to-antibiotics. IRB approval required. Results needed for FDA submission."),
    ("asst", "Shadow mode: predictions logged but not displayed; clinician decisions captured independently. Primary endpoint: time-to-first-antibiotic-administration. Power calculation: n=400 sepsis episodes for 80% power (alpha=0.05, expected 45min improvement). IRB protocol drafted; FDA pre-submission meeting recommended before 510(k) filing."),
],

"Project_Mgmt_Jira": [
    ("user", "Set up a Jira-based project management system for a 25-person engineering team. 3 squads: backend, frontend, infra. 2-week sprints. Need velocity tracking, dependency management across squads, and risk flagging."),
    ("asst", "Jira Software Cloud: 3 team-managed projects linked under one programme board. Epic→Story→Subtask hierarchy. Velocity chart per squad (rolling 6-sprint average). Cross-squad dependencies flagged via 'blocked by' link type with auto-notification."),
    ("user", "Definition of Done: code review approved, unit tests >80% coverage, integration tests pass, security scan clean (SAST via Snyk), product owner sign-off. Automate DoD checks in CI before Jira transition to Done."),
    ("asst", "Jira automation rule: on PR merge, trigger GitHub Actions webhook → check coverage report (≥80%), Snyk scan (0 high/critical), test suite pass → if all pass, auto-transition to In Review; PO must manually approve to Done. Failure posts comment on Jira ticket."),
    ("user", "Sprint health metrics needed: sprint burndown, scope creep rate (<10% target), bug injection rate, lead time vs cycle time. OKR alignment: each epic must map to a Q-level OKR. Escalation path for blocked tickets > 2 business days."),
    ("asst", "Advanced Roadmaps for OKR→Epic mapping. Scope creep tracked as story points added after sprint start / committed points. Jira automation: ticket blocked >2 days → Slack alert to squad lead; >4 days → escalate to EM. Bug injection rate = bugs filed during sprint / stories completed."),
    ("user", "Capacity planning for next quarter: each engineer has variable availability (leave, onboarding). Story point allocation must account for 20% buffer for unplanned work. Integration with HR system (Workday) for leave data."),
    ("asst", "Workday REST API integration: fetch approved leave calendar weekly → compute available days per engineer → convert to story points (1 day = 1.5 SP heuristic). Quarterly capacity = sum(available SP) × 0.80 buffer factor. Advanced Roadmaps team capacity field auto-populated."),
],

}


# ── Dynamic Dialogue Generator ────────────────────────────────────────────

DYNAMIC_DIALOGUE_TOPICS = [
    "Kubernetes multi-cluster federation with service mesh (Istio) and GitOps (ArgoCD), 20 turns",
    "Federated learning system for privacy-preserving NLP across hospital networks, 20 turns",
    "High-frequency trading engine with FPGA tick processing and sub-microsecond latency, 20 turns",
    "MLOps platform design with feature store, model registry, and A/B testing framework, 20 turns",
    "Distributed graph database for fraud detection in real-time payment networks, 20 turns",
    "Autonomous drone fleet management with collision avoidance and BVLOS compliance, 20 turns",
    "LLM fine-tuning pipeline with RLHF, constitutional AI, and red-team evaluation, 20 turns",
    "Edge AI deployment on ARM Cortex-M microcontrollers with TensorFlow Lite, 20 turns",
]

DIALOGUE_GEN_SYSTEM = """You are a senior technical architect generating \
realistic multi-turn dialogues.

Generate a technical dialogue between a USER (junior engineer) and \
ASSISTANT (senior architect). The dialogue MUST contain:
  - At least 5 explicit technical constraints (hard limits)
  - At least 5 explicit decisions (option A chosen over B, with justification)
  - At least 3 causal dependency chains of depth >= 2
  - Iterative refinement: at least 2 earlier constraints revised in later turns
  - Technical specificity: real library names, real version numbers, real thresholds

Format as JSON array: [{"role": "user"/"assistant", "content": "..."}, ...]
Return ONLY the JSON array. No markdown, no explanation.
"""


def generate_dynamic_dialogue(
    topic : str,
    pool  : GroqClientPool,
) -> List[Tuple[str, str]]:
    """
    Generate a synthetic technical dialogue on the given topic via Groq.

    Parameters
    ----------
    topic : description of the dialogue topic and desired turn count
    pool  : GroqClientPool for API calls

    Returns
    -------
    List of (role, text) tuples
    """
    print(f"  Generating dialogue: {topic[:60]}...")
    raw = generate_llm_response(
        DIALOGUE_GEN_SYSTEM,
        f"Generate the dialogue for this topic: {topic}",
        pool,
        max_tokens=3000,
    )
    try:
        jm = re.search(r'\[.*\]', raw, re.DOTALL)
        if jm:
            turns = json.loads(jm.group())
            return [
                (t["role"][:4], t["content"])
                for t in turns
                if "role" in t and "content" in t
            ]
    except Exception:
        pass
    return []


def get_all_dialogues() -> Dict[str, List[Tuple[str, str]]]:
    """Return all 7 hardcoded dialogues."""
    return HARDCODED_DIALOGUES
