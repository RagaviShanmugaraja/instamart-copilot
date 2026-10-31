
# Instamart Customer Intelligence & Support Copilot
## Overview
The Instamart Customer Intelligence & Support Copilot is a data-driven project designed to analyse customer behaviour, identify customers at risk of leaving, and improve customer support operations.

The project uses Python, Pandas, NumPy, SQLite, Scikit-learn and machine learning techniques to build customer analytics and intelligent support workflows.

### Project Objectives
Explore, clean and validate customer, product, order and support ticket datasets.

Analyse sales performance using SQL and Pandas.

Segment customers using RFM analysis and K-means clustering.

Predict customer lapse using machine learning models.

Build a support ticket triage system.

Develop a policy-based RAG assistant and a mock tool-calling refund agent.

### Technologies Used
Python

Pandas and NumPy

SQLite

Scikit-learn

Matplotlib

TF-IDF

Joblib

VS Code

## Reproduce
1. Clone the repository

git clone https://github.com/RagaviShanmugaraja/instamart-copilot.git

cd instamart-copilot
2. Create a virtual environment

python -m venv .venv
3. Install dependencies

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
4. Generate the datasets

.\.venv\Scripts\python.exe generate_data.py
5. Run the data preparation and analysis scripts

.\.venv\Scripts\python.exe audit_data.py

.\.venv\Scripts\python.exe clean_customers.py

.\.venv\Scripts\python.exe clean_orders.py

.\.venv\Scripts\python.exe store_data.py

.\.venv\Scripts\python.exe validate_data.py

.\.venv\Scripts\python.exe run_sql.py

.\.venv\Scripts\python.exe pandas_analysis.py
6. Run Part B – Customer Intelligence
.\.venv\Scripts\python.exe rfm_analysis.py
.\.venv\Scripts\python.exe customer_clustering.py
.\.venv\Scripts\python.exe lapse_prediction.py
7. Run Part C – Support Ticket Triage
.\.venv\Scripts\python.exe triage.py
8. Run Part D – Policy RAG and Refund Agent
The policy retrieval functions are implemented in policy_rag.py and are used by the refund agent.

.\.venv\Scripts\python.exe -c "from refund_agent import run_all_scenarios; run_all_scenarios()"
.\.venv\Scripts\python.exe refund_agent.py
The first command regenerates the seven scenario results in outputs/agent_runs.jsonl. The second command runs the checkpoint-and-resume demonstration and writes its checkpoint under state/.

Run commands from the repository root. The supplied generate_data.py and mock_llm.py files should remain unmodified.

## Part A – Data Foundation
### 1. Data Exploration and Audit
The raw datasets were explored to understand their structure, missing values, duplicate records and data quality issues.

The datasets included:

Customers

Products

Orders

Support tickets

Policy evaluation data

Agent scenarios

Important audit findings:

| Data Quality Check | Result |

|---|---:|

| Raw order rows | 13,084 |

| Exact duplicate order rows | 119 |

| Pending orders | 656 |

| Distinct order IDs | 12,309 |

| Non-numeric order amounts | 120 |

| Negative delivery times | 106 |

| Blank delivery times | 2,541 |

| Customers with blank ages | 61 |

| Customers with blank genders | 30 |

### 2. Data Cleaning
#### Customer data:

Standardised city names.

Normalised gender values into Male, Female and Unknown.

Converted age values to numeric.

Treated ages outside the 10–100 range as missing.

Filled missing ages using the median age for each city.

Converted signup dates into datetime format.

#### Order data:

Converted order timestamps into datetime format.

Removed 119 exact duplicate rows.

Removed 656 duplicate order records, retaining the latest record per order ID.

Filled missing order amounts using product price, quantity and discount.

Corrected negative delivery times.

Filled missing delivery times for completed orders using the product category median.

Preserved missing delivery times for cancelled orders.

#### Final cleaned data:

| Dataset | Rows |

|---|---:|

| Customers | 1,200 |

| Products | 40 |

| Orders | 12,309 |

Completed orders: 11,110

Cancelled orders: 1,199

Completed order revenue: INR 2,998,668.80

### 3. SQLite Database
The cleaned datasets were stored in an SQLite database named:

data/instamart.db

The database contains three tables:

customers

products

orders

### 4. SQL Analysis
Seven SQL queries were implemented and verified.

| Query | Analysis |

|---|---|

| Q1 | High-value customers based on completed order spending |

| Q2 | Revenue by city and product category |

| Q3 | Products priced above the catalogue average |

| Q4 | Customers without completed orders |

| Q5 | Customers spending above the average |

| Q6 | Customer ranking within each city using window functions |

| Q7 | Daily revenue with a running total |

#### Important SQL results:

Completed order revenue: INR 2,998,668.80.

Customers with above-average spending: 550.

Customers without completed orders: 25.

Customers returned by the city-ranking query: 1,175.

Daily revenue records: 183.

Final running revenue total: INR 2,998,668.80.

### 5. Pandas Analysis
Four additional analyses were performed using Pandas.

#### Revenue by city and category

Used pivot_table() to calculate revenue across cities and product categories.

Total revenue: INR 2,998,668.80.

Hyderabad recorded the highest revenue at INR 528,589.90.

#### City-level sales performance

Used groupby() and aggregation.

Hyderabad recorded the highest revenue.

Delhi recorded the highest number of completed orders.

Mumbai recorded the highest average order value.

#### Order amount distribution and outliers

| Metric | Value |

|---|---:|

| Mean | INR 269.91 |

| Median | INR 135.00 |

| Standard deviation | 432.07 |

| Skewness | 4.61 |

| Q1 | INR 64.80 |

| Q3 | INR 297.00 |

| IQR | INR 232.20 |

| Upper outliers | 897 |

#### Correlation analysis

Order amount and discount percentage: -0.0951.

Order amount and quantity: approximately +0.087.

Order amount and delivery time: approximately -0.003.

The correlations were weak, indicating no strong linear relationship between these variables and order amount.

### 6. Data Validation
The cleaned datasets and database were checked against the project validation requirements.

Validation result: ALL CHECKPOINTS PASSED

## Part B – Customer Intelligence
### 1. RFM Analysis
RFM analysis was performed using completed orders only.

Snapshot date: 2025-10-01.

Customers analysed: 1,175.

#### RFM metrics:

Recency: Days since the customer's most recent completed order.

Frequency: Number of completed orders.

Monetary: Total spending on completed orders.

| Metric | Mean | Minimum | Maximum |

|---|---:|---:|---:|

| Recency (days) | 34.55 | 1 | 181 |

| Frequency | 9.46 | 1 | 33 |

| Monetary (INR) | 2,552.06 | 12.00 | 14,766.55 |

Customers were assigned RFM scores and grouped into six segments.

| Customer Segment | Count |

|---|---:|

| Lost Customers | 328 |

| Champions | 235 |

| Others | 212 |

| Loyal Customers | 196 |

| At Risk | 142 |

| Potential Loyalists | 62 |

| Total | 1,175 |

### 2. K-means Customer Clustering
K-means clustering was performed using Recency, Frequency and Monetary features. StandardScaler was used to scale the features.

The elbow method and silhouette scores were evaluated for K=2 to K=8.

| K | Inertia | Silhouette Score |

|---:|---:|---:|

| 2 | 1820.65 | 0.4650 |

| 3 | 1268.07 | 0.4452 |

| 4 | 742.93 | 0.5256 |

| 5 | 617.35 | 0.4805 |

| 6 | 503.14 | 0.4825 |

| 7 | 408.45 | 0.4164 |

| 8 | 366.51 | 0.4123 |

Selected K = 4.

K=4 achieved the highest silhouette score of 0.5256. The inertia also showed a substantial reduction between K=3 and K=4. Together, these results support selecting four clusters.

#### Cluster summary:

| Cluster | Customers | Avg. Recency | Avg. Frequency | Avg. Monetary (INR) |

|---:|---:|---:|---:|---:|

| 0 | 436 | 29.85 | 3.77 | 1,057.62 |

| 1 | 138 | 25.09 | 6.77 | 7,129.70 |

| 2 | 419 | 9.75 | 19.40 | 3,410.02 |

| 3 | 182 | 110.04 | 2.20 | 685.99 |

#### Business segment names:

Cluster 0: Occasional Customers

Cluster 1: Premium Customers

Cluster 2: Frequent & Recent Customers

Cluster 3: Inactive Customers

The Premium Customers segment contains 138 customers and has the highest average monetary value.

### 3. PCA Analysis
Principal Component Analysis was performed on the scaled RFM features using two components.

| Component | Explained Variance |

|---|---:|

| PC1 | 63.43% |

| PC2 | 21.80% |

| Total | 85.23% |

The first two principal components retain 85.23% of the total variance, allowing the three-dimensional RFM data to be represented in two dimensions while preserving most of its information.

### 4. Customer Lapse Prediction
Customer lapse was defined using the following rules:

Cohort: Customers with at least one completed order before 2025-09-01.

Lapsed (1): No completed orders between 2025-09-01 and 2025-09-30.

Not lapsed (0): At least one completed order during September 2025.

#### Cohort results:

| Metric | Result |

|---|---:|

| Total customers | 1,156 |

| Lapsed customers | 443 |

| Not-lapsed customers | 713 |

| Lapse rate | 38.32% |

| Training rows | 924 |

| Testing rows | 232 |

The train-test split used an 80:20 ratio, stratified by the target, with random_state=42.

Features were calculated using historical data before the cutoff date and the clean customer table. These included recency, frequency, monetary, average discount, cancellation rate, average delivery time, favourite product category, city, age, gender and customer tenure.

OneHotEncoder was used for categorical features, and StandardScaler was used for numerical features. Preprocessing was fitted on training data only.

The learned training frequency mean was 8.1147, compared with the full cohort mean of 8.0285. The difference occurs because the training set contains only a subset of the cohort.

No feature values were missing.

### 5. Model Training and Evaluation
A DummyClassifier was used as the baseline.

| Model | Accuracy | Lapsed F1 |

|---|---:|---:|

| Baseline | 61.64% | 0.0000 |

Logistic Regression and Random Forest were tuned using GridSearchCV with five-fold cross-validation and F1 scoring.

#### Best parameters:

Logistic Regression: C=100.

Random Forest: max_depth=8, min_samples_leaf=10.

| Model | Best CV F1 | Test Precision | Test Recall | Test F1 |

|---|---:|---:|---:|---:|

| Logistic Regression | 0.6506 | 0.6444 | 0.6517 | 0.6480 |

| Random Forest | 0.6811 | 0.6591 | 0.6517 | 0.6554 |

#### Test confusion matrices:

Logistic Regression


[[111, 32],

 [ 31, 58]]
Random Forest


[[113, 30],

 [ 31, 58]]
Random Forest was selected because it achieved the higher test F1 score and higher precision, while recall was the same as Logistic Regression.

### 6. Class-Balancing Experiment
The Logistic Regression training data was balanced by randomly oversampling the lapsed class with replacement. Oversampling was performed only on the training data.

| Metric | Original Logistic Regression | Balanced Logistic Regression |

|---|---:|---:|

| Precision | 0.6444 | 0.5891 |

| Recall | 0.6517 | 0.8539 |

| F1 Score | 0.6480 | 0.6972 |

The balanced model improved recall and F1. It identified more lapsed customers, but precision decreased, resulting in more active customers being incorrectly flagged.

### 7. Feature Importance and Interpretation
The five most important Random Forest features were:

| Feature | Importance |

|---|---:|

| Frequency | 0.2796 |

| Monetary | 0.2712 |

| Recency | 0.1151 |

| Average Delivery Minutes | 0.0613 |

| Cancellation Rate | 0.0523 |

#### Logistic Regression coefficient interpretations:

Recency: +0.0396 — higher recency is associated with increased lapse odds.

Frequency: -1.6982 — higher purchase frequency is associated with reduced lapse odds.

Cancellation Rate: -0.1236 — a negative association in this fitted model, conditional on the other features.

These are model associations and should not be interpreted as proof of causation.

### 8. Model Persistence and Retention List
The selected Random Forest model was saved using joblib:

outputs/lapse_model.joblib

The saved model was reloaded, and predictions were compared with the original model predictions.

Verification result: Predictions identical after reload: True.

A retention list was generated by ranking premium customers in the test set according to predicted lapse probability.

| Metric | Result |

|---|---:|

| Premium customers in test set | 40 |

| Customers in retention list | 10 |

| Truly lapsed customers in top 10 | 1 |

The list was saved to:

outputs/retention_list.csv

Limitation: Premium customer labels were assigned using full-window RFM data, including September 2025. Therefore, the retention list is illustrative and is not a leakage-free evaluation. The lapse prediction model's features were still restricted to information available before the September cutoff.

## Part C – Support Ticket Triage with Guardrails
### 1. Overview
A support ticket triage pipeline was developed using the supplied offline mock_llm.py. The pipeline processes 164 support tickets with privacy protection, input guardrails, retry handling, response validation, token budgeting and customer-based routing.

The supplied mock LLM was used without modification. No external model API or network connection was required.

### 2. Input Guardrails
Tickets were checked before any model call.

Empty or whitespace-only tickets were assigned skipped_empty.

Tickets exceeding 400 estimated tokens were assigned flagged_long.

Both groups were excluded from model calls.

| Status | Count |

|---|---:|

| Eligible | 156 |

| Skipped empty | 4 |

| Flagged long | 4 |

| Total | 164 |

All four long tickets estimated to 734 tokens.

### 3. Personal Information Redaction
Three regular expressions were applied in the required order before constructing model prompts.

| Information | Placeholder | Substitutions |

|---|---|---:|

| Email addresses | [EMAIL] | 55 |

| Phone numbers | [PHONE] | 91 |

| Order IDs | [ORDER_ID] | 132 |

| Total | | 278 |

A verification check reapplied all three patterns to the redacted text.

Result: Zero remaining email addresses, phone numbers and order IDs.

The user message content sent to the mock model consisted exactly of the redacted ticket text.

### 4. Classification Prompts
Two system prompts were implemented:

Zero-shot prompt: classification instructions without examples.

Few-shot prompt: classification instructions with labelled billing, delivery and fraud examples.

Both prompts required JSON-only responses containing a category and reason, with four allowed categories: billing, delivery, technical and fraud.

### 5. Retry Handling
Every model call was wrapped in a retry function.

Maximum retries: 3.

Base delay: 0.05 seconds.

Exponential backoff with random jitter.

One log entry for each retry, including ticket ID, retry number and wait time.

Across each full run, 19 transient failures occurred. All 19 transient failures were recovered on the first retry, with no permanent failures.

### 6. Response Validation
Every model response was validated.

Valid JSON with an allowed category: classified_json.

Non-JSON response containing exactly one allowed category word: classified_keyword.

Invalid JSON, invalid category or ambiguous response: flagged_invalid_output.

The keyword extraction used re.findall(r"[a-z]+", text.lower()) to correctly handle hyphenated words.

Final status counts for each full run:

| Status | Count |

|---|---:|

| classified_json | 135 |

| classified_keyword | 12 |

| flagged_invalid_output | 9 |

| flagged_long | 4 |

| skipped_empty | 4 |

| Total | 164 |

### 7. Zero-shot vs Few-shot Comparison
Both approaches classified 147 tickets.

| Metric | Zero-shot | Few-shot |

|---|---:|---:|

| Classified tickets | 147 | 147 |

| Accuracy | 85.71% | 100.00% |

| Total tokens | 12,364 | 21,100 |

| Tokens per classified ticket | 84.11 | 143.54 |

The zero-shot confusion matrix was:

| Actual / Predicted | Billing | Delivery | Technical | Fraud |

|---|---:|---:|---:|---:|

| Billing | 38 | 0 | 0 | 0 |

| Delivery | 0 | 36 | 0 | 0 |

| Technical | 10 | 0 | 29 | 0 |

| Fraud | 11 | 0 | 0 | 23 |

The few-shot prompt required 8,736 additional tokens, approximately 70.6% more than zero-shot. However, accuracy improved from 85.71% to 100%. The additional token cost is justified when accurate classification, particularly of fraud and technical tickets, is important.

### 8. Token Budget Demonstration
The zero-shot pipeline was executed with a token budget of 6,000.

| Metric | Result |

|---|---:|

| Configured token budget | 6,000 |

| Actual tokens used | 6,038 |

| Classified tickets | 72 |

| Budget-exhausted tickets | 80 |

The budget is checked before each model call. Therefore, the final permitted call can take the running total above the configured budget.

Output: outputs/triage_budget_demo.csv

### 9. Customer-Based Routing
The final few-shot pipeline used a token budget of 100,000 and customer segment information from Part B.

Routing rules:

Empty, long, invalid-output and budget-exhausted tickets were routed to human review with NORMAL priority.

Fraud tickets were routed to the security team with HIGH priority.

Premium customers were routed to the premium desk with HIGH priority.

Other classified tickets were routed to the support queue with NORMAL priority.

Customers without a segment record were treated as non-premium.

Final routing results:

| Route | Tickets |

|---|---:|

| support_queue | 102 |

| security_team | 34 |

| human_review | 17 |

| premium_desk | 11 |

| Total | 164 |

Priority distribution:

| Priority | Tickets |

|---|---:|

| HIGH | 45 |

| NORMAL | 119 |

Output: outputs/triage_results.csv

### 10. Validation and Safety
Automated assertions verified:

All 164 tickets were retained.

Ticket IDs were unique.

Classification status counts matched the expected results.

Human-review tickets followed the required routing rules.

Fraud tickets were routed to the security team.

Priority assignments were consistent with routing rules.

Final token usage matched the recorded total.

Final validation result: ALL FINAL TRIAGE VALIDATION CHECKS PASSED.

### 11. Limitations
The offline mock LLM is deterministic and does not represent the full behaviour of a production language model.

The classification results are based on the supplied mock's keyword logic.

The retry mechanism handles simulated transient errors; real API failures may require additional error handling.

Token estimation uses the mock's word-based approximation rather than a production tokenizer.

Human review is required for invalid, empty, long and budget-exhausted tickets.

## Part D – Policy RAG and Customer Support Agent
### Task 1 – Policy Document Chunking
The five policy documents were loaded from the data/policies/ directory and divided into smaller chunks using a fixed-size chunking strategy.

Configuration:

Chunk size: 40 words

Overlap: 10 words

Step size: 30 words

The overlap ensures that important information near chunk boundaries is not completely lost.

Chunking results:

| Policy Document | Number of Chunks |

|---|---:|

| refund_policy.txt | 4 |

| delivery_policy.txt | 3 |

| escalation_policy.txt | 3 |

| payment_policy.txt | 3 |

| privacy_policy.txt | 2 |

| Total | 15 |

The chunks were stored with their corresponding source filenames to maintain traceability.

### Task 2 – TF-IDF Retrieval
TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert policy chunks into numerical vectors.

Term Frequency (TF): Measures how frequently a word appears in a document.

Inverse Document Frequency (IDF): Gives more importance to rare words and less importance to commonly occurring words.

The TF-IDF vectorizer was fitted on all policy chunks. Cosine similarity was then used to compare the user's query with the stored vectors.

The retrieve(query, k=3) function returns the three most relevant policy chunks, including their source filenames, similarity scores, and text.

Test query: refund policy late delivery

The top three retrieved chunks belonged to refund_policy.txt.

Limitation: TF-IDF mainly relies on keyword matching. It may not retrieve semantically similar information when different words are used to express the same meaning.

### Task 3 – Policy Question Answering
An answer(question) function was implemented to generate answers using the retrieved policy information.

Workflow:

Receive a user question.

Retrieve the top three relevant policy chunks.

Construct a context containing the source filenames and chunk text.

Send the context and question to the mock LLM.

Return the generated answer along with the source documents.

The system prompt begins with the required POLICY ASSISTANT marker. The user message follows the required format containing CONTEXT and QUESTION.

The assistant is instructed to answer using only the supplied policy context.

If the required information is unavailable, the assistant returns:

I could not find the information in the supplied policies.

Example test question:

What percentage of the order amount is refunded when delivery takes more than 45 minutes?

Result: The answer correctly identified that the customer receives a full refund of the order amount.

Retrieved sources included refund_policy.txt and delivery_policy.txt.

### Task 4 – Policy Evaluation
The RAG system was evaluated using all 14 questions in policy_eval.csv.

The evaluation checked:

Whether the expected policy document appeared in the top three retrieved chunks.

Whether the generated answer contained the expected fact.

Whether unanswerable questions correctly triggered the fallback response.

Results with 10-word overlap:

| Evaluation Metric | Result |

|---|---:|

| Total questions | 14 |

| Answerable questions | 12 |

| Unanswerable questions | 2 |

| Expected document retrieved in top 3 | 12/12 |

| Expected facts correctly answered | 12/12 |

| Unanswerable questions handled with fallback | 2/2 |

Comparison without overlap:

| Evaluation Metric | Result |

|---|---:|

| Expected document retrieved in top 3 | 12/12 |

| Expected facts correctly answered | 10/12 |

| Unanswerable questions handled with fallback | 2/2 |

The overlap-based approach achieved better answer accuracy because it helped preserve information that could otherwise be separated across chunk boundaries.

### Task 5 – Tool Definitions and Refund Ledger
Five tools were implemented to allow the support agent to interact with the order database and policy documents.

| Tool | Purpose |

|---|---|

| get_order | Retrieves order details from SQLite |

| search_policy | Searches policy documents using TF-IDF |

| calculate_refund | Calculates the eligible refund amount |

| issue_refund | Records an approved refund |

| escalate_to_human | Records a refund requiring human approval |

The tools were registered using TOOLS_SPEC and mapped to their corresponding Python functions through TOOL_MAP.

A JSONL refund ledger was implemented to record refund-related actions.

Each ledger entry contains:

Timestamp

Tool name

Order ID

Refund amount in INR

This provides a record of refund and escalation actions.

### Task 6 – ReAct Agent Implementation
A ReAct (Reasoning and Acting) agent was implemented using the mock LLM.

The agent follows an iterative process:

Receive a customer request.

Analyse the request and select an appropriate tool.

Execute the selected tool.

Observe the tool result.

Decide the next action.

Return a final response when the task is complete.

The agent was configured with a maximum of 6 model iterations.

Seven predefined customer support scenarios were executed.

Scenario results:

| Scenario | Expected Outcome | Actual Outcome |

|---|---|---|

| S1 – Late delivery refund | REFUND – INR 128.25 | REFUND |

| S2 – Late delivery refund | REFUND – INR 135.00 | REFUND |

| S3 – High-value refund | ESCALATE – INR 1,999 | ESCALATE |

| S4 – Delivery within allowed time | DENIED | DENIED |

| S5 – Cancelled/ineligible order | DENIED | DENIED |

| S6 – Invalid order ID | NOT_FOUND | NOT_FOUND |

| S7 – Repeated tool-call loop | FALLBACK_MAX_ITERATIONS | FALLBACK_MAX_ITERATIONS |

The agent's results, model calls, token usage, and individual tool execution steps were saved in outputs/agent_runs.jsonl.

### Task 7 – Safety Limits
Safety controls were implemented to prevent the agent from running indefinitely or consuming excessive resources.

Three limits were configured:

Maximum iterations: 6 model calls.

Token limit: 6,000 tokens.

Time limit: 10 seconds.

The agent returns a human-support fallback response when a safety limit is reached.

Safety test results:

| Test | Configuration | Result |

|---|---|---|

| Maximum iteration test | 6 iterations | FALLBACK_MAX_ITERATIONS |

| Token limit test | 200 tokens | FALLBACK_TOKEN_LIMIT |

| Timeout test | 0 seconds | FALLBACK_TIMEOUT |

The token-limit test stopped the agent before executing another tool after the token budget was exceeded.

These controls help ensure that the agent operates within predefined resource limits.

### Task 8 – Checkpointing and Resume
Checkpointing was implemented to preserve the agent's execution state after every tool observation.

Each checkpoint stores:

Run ID

Scenario ID

Conversation messages

Completed execution steps

Model call count

Tokens used

Iteration count

Tool execution history

Checkpoint files are stored in the state/ directory in JSON format.

A resume_agent(run_id) function was implemented to load a saved checkpoint and continue execution from the previous state.

Crash and resume demonstration:

The agent executed get_order.

The agent executed search_policy.

A simulated crash occurred after the second tool execution.

The saved checkpoint was loaded.

The agent resumed and executed calculate_refund.

The agent executed issue_refund.

The agent returned the final refund response.

Final result:

| Metric | Result |

|---|---|

| Status | REFUND |

| Refund amount | INR 128.25 |

| Total tool executions | 4 |

| Repeated tool executions | 0 |

| Model calls | 5 |

| Tokens used | 1,022 |

The demonstration confirmed that the agent resumed successfully without repeating previously completed tools.

### Task 9 – ReAct vs Plan-and-Execute
ReAct was selected for the Instamart Customer Intelligence & Support Copilot because it allows the agent to reason, execute a tool, and observe the result before deciding the next action.

In the ReAct implementation, the agent uses multiple model calls depending on the scenario. For example, S1 required 5 model calls and 1,022 tokens to complete the refund process. Other scenarios required between 2 and 6 model calls, depending on the complexity of the request.

Plan-and-Execute would first generate a complete sequence of actions and then execute the planned steps. This approach can be useful for predictable workflows, but it may require replanning when an unexpected observation occurs.

ReAct handles unexpected observations more naturally because the agent can use each tool result to decide its next action. For example, it can identify an order that does not exist, deny an ineligible refund, or escalate a high-value refund.

Although ReAct can require multiple model calls and additional tokens, it provides better flexibility for customer support scenarios where order status, delivery time, and refund eligibility can vary.

Therefore, ReAct was chosen because this project requires dynamic decisions based on tool observations rather than a fixed sequence of actions.

### Conclusion
Part D successfully implemented a policy-based Retrieval-Augmented Generation (RAG) system and a ReAct customer support agent.

The system can retrieve relevant policy information, answer customer questions, execute refund-related tools, handle different customer scenarios, enforce safety limits, and recover from simulated crashes using checkpointing.

All nine implementation tasks were completed using the supplied mock LLM and local project data, without requiring paid external APIs.

## Limitations and Next Steps
### Limitations
The project uses the supplied offline mock LLM rather than a production language model.
TF-IDF retrieval relies mainly on keyword matching and may miss semantically similar wording.
Customer lapse predictions are based on historical data and do not guarantee future customer behaviour.
The retention list is illustrative; premium-customer labels use full-window RFM data, including September 2025, as explained in Part B.
Refund and escalation actions are recorded in a local JSONL ledger. They are not connected to a real payment gateway or production order-management system.
The mock LLM and local datasets do not represent all conditions that would occur in a live customer-support environment.
### Next Steps
Integrate a production-grade language model with appropriate privacy and security controls.
Improve policy retrieval using semantic embeddings or a hybrid retrieval approach.
Connect the agent to a secure, real-time order-management system.
Add an approval workflow and safeguards before processing real refunds.
Monitor model performance, retrieval quality and customer-support outcomes using new data.
Add automated tests and monitoring for failures, latency, token usage and tool execution.
