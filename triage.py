
import pandas as pd
import re
import json
import random
import time

import mock_llm


# CONFIGURATION
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05

ALLOWED_CATEGORIES = [
    "billing",
    "delivery",
    "technical",
    "fraud"
]


# LOAD TICKETS
tickets = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)

print("TOTAL TICKETS:", len(tickets))


# INPUT GUARDRAILS
def check_ticket(text):

    if not text.strip():
        return "skipped_empty"

    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"

    return "eligible"


tickets["input_status"] = tickets["text"].apply(check_ticket)

print("\nINPUT GUARDRAIL RESULTS:")
print(tickets["input_status"].value_counts())

print("\nTOKEN ESTIMATION FOR LONG TICKETS:")

long_tickets = tickets[
    tickets["input_status"] == "flagged_long"
]

for _, ticket in long_tickets.iterrows():
    print(
        ticket["ticket_id"],
        mock_llm.est_tokens(ticket["text"])
    )


# PERSONAL INFORMATION REDACTION

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def redact_text(text):

    email_count = 0
    phone_count = 0
    order_count = 0

    # 1. Redact emails
    text, email_count = re.subn(
        EMAIL_PATTERN,
        "[EMAIL]",
        text
    )

    # 2. Redact phone numbers
    text, phone_count = re.subn(
        PHONE_PATTERN,
        "[PHONE]",
        text
    )

    # 3. Redact order IDs
    text, order_count = re.subn(
        ORDER_PATTERN,
        "[ORDER_ID]",
        text
    )

    return text, email_count, phone_count, order_count


# APPLY REDACTION TO ALL 164 TICKETS

redacted_texts = []

total_emails = 0
total_phones = 0
total_orders = 0

for text in tickets["text"]:

    redacted, emails, phones, orders = redact_text(text)

    redacted_texts.append(redacted)

    total_emails += emails
    total_phones += phones
    total_orders += orders


tickets["redacted_text"] = redacted_texts


# REDACTION SUMMARY

print("\nREDACTION RESULTS:")
print("Emails redacted:", total_emails)
print("Phone numbers redacted:", total_phones)
print("Order IDs redacted:", total_orders)

total_redactions = total_emails + total_phones + total_orders

print("Total redactions:", total_redactions)


# VERIFY NO PERSONAL INFORMATION REMAINS

remaining_emails = sum(
    len(re.findall(EMAIL_PATTERN, text))
    for text in tickets["redacted_text"]
)

remaining_phones = sum(
    len(re.findall(PHONE_PATTERN, text))
    for text in tickets["redacted_text"]
)

remaining_orders = sum(
    len(re.findall(ORDER_PATTERN, text))
    for text in tickets["redacted_text"]
)

print("\nREDACTION VERIFICATION:")
print("Remaining emails:", remaining_emails)
print("Remaining phones:", remaining_phones)
print("Remaining order IDs:", remaining_orders)

assert remaining_emails == 0
assert remaining_phones == 0
assert remaining_orders == 0

print("PRIVACY CHECK PASSED!")


# ZERO-SHOT SYSTEM PROMPT

ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return a JSON-only response in this exact format:
{"category": "...", "reason": "..."}

Do not include explanations outside the JSON response.
"""


# FEW-SHOT SYSTEM PROMPT

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return a JSON-only response in this exact format:
{"category": "...", "reason": "..."}

Do not include explanations outside the JSON response.

EXAMPLES:

Ticket: I was charged twice for my order.
Response: {"category": "billing", "reason": "Duplicate payment"}

Ticket: My parcel has not arrived yet.
Response: {"category": "delivery", "reason": "Parcel not received"}

Ticket: Someone accessed my account without permission.
Response: {"category": "fraud", "reason": "Unauthorized account access"}
"""


# VERIFY PROMPTS

assert "CLASSIFY" in ZERO_SHOT_PROMPT
assert "CLASSIFY" in FEW_SHOT_PROMPT

assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
assert "EXAMPLES:" in FEW_SHOT_PROMPT

for category in ALLOWED_CATEGORIES:
    assert category in ZERO_SHOT_PROMPT
    assert category in FEW_SHOT_PROMPT

print("\nPROMPT VALIDATION PASSED!")
print("Zero-shot prompt is valid.")
print("Few-shot prompt is valid.")


# RETRY CONFIGURATION

MAX_RETRIES = 3


def call_with_retry(messages, ticket_id):

    for retry_number in range(MAX_RETRIES + 1):

        try:
            response = mock_llm.chat(messages)
            return response

        except mock_llm.TransientLLMError:

            if retry_number == MAX_RETRIES:
                print(
                    f"{ticket_id}: Maximum retries reached."
                )
                raise

            wait_time = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY | Ticket: {ticket_id} | "
                f"Retry number: {retry_number} | "
                f"Wait: {wait_time:.4f} seconds"
            )

            time.sleep(wait_time)


print("\nRETRY FUNCTION CREATED SUCCESSFULLY!")


# MODEL RESPONSE VALIDATION

def validate_response(response_text):

    # First, try to parse JSON
    try:
        result = json.loads(response_text)

        category = result.get("category")

        if category in ALLOWED_CATEGORIES:
            return category, "classified_json"

        # Valid JSON but category is not allowed
        return None, "flagged_invalid_output"

    except (json.JSONDecodeError, AttributeError, TypeError):

        # Extract lowercase words using regex
        words = re.findall(
            r"[a-z]+",
            response_text.lower()
        )

        # Find allowed categories in the extracted words
        found_categories = [
            word for word in words
            if word in ALLOWED_CATEGORIES
        ]

        # Accept only if exactly one category is present
        if len(found_categories) == 1:
            return found_categories[0], "classified_keyword"

        return None, "flagged_invalid_output"


print("\nRESPONSE VALIDATION FUNCTION CREATED!")


# VALIDATION TESTS

test_responses = [
    '{"category": "billing", "reason": "Payment issue"}',
    'The customer has a billing-related problem.',
    '{"category": "shipping", "reason": "Parcel issue"}',
    'The customer has a billing and fraud problem.',
    'I cannot classify this ticket.'
]

print("\nVALIDATION TEST RESULTS:")

for response_text in test_responses:

    category, status = validate_response(response_text)

    print(
        f"Response: {response_text}\n"
        f"Category: {category}, Status: {status}\n"
    )


# FULL SUPPORT TICKET TRIAGE PIPELINE

from sklearn.metrics import confusion_matrix, accuracy_score


def run_pipeline(system_prompt, prompt_name, token_budget=None):

    # Reset mock immediately before each independent run
    mock_llm.reset()

    results = []
    running_tokens = 0

    print(f"\nSTARTING {prompt_name.upper()} RUN")

    # Process tickets in original file order
    for _, ticket in tickets.iterrows():

        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket["text"]
        input_status = ticket["input_status"]

        category = None
        status = input_status
        tokens_used = 0

        # Skip empty and long tickets
        if input_status in ["skipped_empty", "flagged_long"]:

            results.append({
                "ticket_id": ticket_id,
                "customer_id": customer_id,
                "status": status,
                "category": category,
                "true_category": ticket["true_category"],
                "tokens_used": tokens_used
            })

            continue

        # Check token budget before making a model call
        if token_budget is not None and running_tokens >= token_budget:

            status = "budget_exhausted"

        else:

            redacted_text = ticket["redacted_text"]

            # Privacy assertions before sending prompt
            assert re.search(EMAIL_PATTERN, redacted_text) is None
            assert re.search(PHONE_PATTERN, redacted_text) is None
            assert re.search(ORDER_PATTERN, redacted_text) is None

            # User message content must be exactly redacted text
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": redacted_text
                }
            ]

            try:

                response = call_with_retry(
                    messages,
                    ticket_id
                )

                tokens_used = response.usage.total_tokens
                running_tokens += tokens_used

                response_text = response.choices[0].message.content

                category, status = validate_response(
                    response_text
                )

            except mock_llm.TransientLLMError:

                status = "flagged_invalid_output"
                category = None

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "true_category": ticket["true_category"],
            "tokens_used": tokens_used
        })

    results_df = pd.DataFrame(results)

    # Calculate metrics using classified tickets only
    classified = results_df[
        results_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    classified_count = len(classified)

    if classified_count > 0:

        accuracy = accuracy_score(
            classified["true_category"],
            classified["category"]
        )

        tokens_per_ticket = running_tokens / classified_count

    else:

        accuracy = 0
        tokens_per_ticket = 0

    print(f"\n{prompt_name.upper()} RESULTS")
    print("Total tickets:", len(results_df))
    print("Classified tickets:", classified_count)
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", running_tokens)
    print("Tokens per classified ticket:", round(tokens_per_ticket, 2))

    print("\nSTATUS COUNTS:")
    print(results_df["status"].value_counts())

    return results_df, running_tokens


# RUN ZERO-SHOT PIPELINE

zero_shot_results, zero_shot_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    "zero-shot"
)


# RUN FEW-SHOT PIPELINE

few_shot_results, few_shot_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    "few-shot"
)


# ZERO-SHOT CONFUSION TABLE

zero_classified = zero_shot_results[
    zero_shot_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

cm = confusion_matrix(
    zero_classified["true_category"],
    zero_classified["category"],
    labels=ALLOWED_CATEGORIES
)

confusion_df = pd.DataFrame(
    cm,
    index=ALLOWED_CATEGORIES,
    columns=ALLOWED_CATEGORIES
)

print("\nZERO-SHOT CONFUSION TABLE:")
print(confusion_df)


# FEW-SHOT CONFUSION TABLE

few_classified = few_shot_results[
    few_shot_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]

few_accuracy = accuracy_score(
    few_classified["true_category"],
    few_classified["category"]
)

print("\nFEW-SHOT ACCURACY:")
print(round(few_accuracy * 100, 2), "%")

print("\nTASK 6 PIPELINE COMPLETED!")

# TOKEN BUDGET DEMONSTRATION

TOKEN_BUDGET = 6000

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    "zero-shot-budget",
    token_budget=TOKEN_BUDGET
)

# SAVE BUDGET RESULTS

budget_results.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)

print("\nBUDGET DEMONSTRATION:")
print("Token budget:", TOKEN_BUDGET)
print("Tokens used:", budget_tokens)

print("\nBUDGET STATUS COUNTS:")
print(budget_results["status"].value_counts())

print("\nBudget demo saved successfully!")


# FINAL FEW-SHOT TRIAGE WITH CUSTOMER ROUTING

FINAL_TOKEN_BUDGET = 100000

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    "final-few-shot",
    token_budget=FINAL_TOKEN_BUDGET
)


# LOAD CUSTOMER SEGMENTS

segments = pd.read_csv("outputs/customer_segments.csv")

# Keep only the columns needed for routing
segments = segments[
    ["customer_id", "segment_name", "is_premium"]
]


# MERGE TICKETS WITH CUSTOMER SEGMENTS

final_results = final_results.merge(
    segments,
    on="customer_id",
    how="left"
)


# ROUTING FUNCTION

def assign_routing(row):

    human_review_statuses = [
        "skipped_empty",
        "flagged_long",
        "flagged_invalid_output",
        "budget_exhausted"
    ]

    if row["status"] in human_review_statuses:

        return "human_review", "NORMAL"

    elif row["category"] == "fraud":

        return "security_team", "HIGH"

    elif row["is_premium"] == True:

        return "premium_desk", "HIGH"

    else:

        return "support_queue", "NORMAL"


# APPLY ROUTING

routing_results = final_results.apply(
    assign_routing,
    axis=1
)

final_results["route"] = [
    result[0] for result in routing_results
]

final_results["priority"] = [
    result[1] for result in routing_results
]


# FILL MISSING SEGMENT NAMES

final_results["segment_name"] = (
    final_results["segment_name"].fillna("")
)


# SELECT REQUIRED COLUMNS

final_results = final_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]


# SAVE FINAL RESULTS

final_results.to_csv(
    "outputs/triage_results.csv",
    index=False
)


# FINAL VERIFICATION

print("\nFINAL TRIAGE RESULTS")
print("Total rows:", len(final_results))
print("Total tokens:", final_tokens)

print("\nSTATUS COUNTS:")
print(final_results["status"].value_counts())

print("\nROUTING COUNTS:")
print(final_results["route"].value_counts())

print("\nPRIORITY COUNTS:")
print(final_results["priority"].value_counts())

print("\nFINAL CSV SAVED SUCCESSFULLY!")
print("outputs/triage_results.csv")


# FINAL ROUTING VALIDATION

assert len(final_results) == 164
assert final_results["ticket_id"].is_unique

assert final_results["status"].value_counts().get(
    "classified_json", 0
) == 135

assert final_results["status"].value_counts().get(
    "classified_keyword", 0
) == 12

human_statuses = [
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
]

for _, row in final_results.iterrows():

    if row["status"] in human_statuses:
        assert row["route"] == "human_review"
        assert row["priority"] == "NORMAL"

    elif row["category"] == "fraud":
        assert row["route"] == "security_team"
        assert row["priority"] == "HIGH"

    elif row["route"] == "premium_desk":
        assert row["priority"] == "HIGH"

    else:
        assert row["route"] == "support_queue"
        assert row["priority"] == "NORMAL"

assert final_results["route"].value_counts().get(
    "human_review", 0
) == 17

assert final_results["tokens_used"].sum() == 21100

print("\nALL FINAL TRIAGE VALIDATION CHECKS PASSED!")
