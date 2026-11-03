"""Plain-English governance text for the two AI features.

Single source of truth: the dashboard panels and docs/solution_overview.pdf both read from here,
and tests assert that each panel states trigger, false positive, false negative and human-in-the-loop.
"""

ASK_NOTES = {
    "title": "Ask the Portfolio",
    "how": ("An analyst types a question. Guardrails run first. An LLM (or the built-in offline interpreter) turns the question into a small "
            "JSON query plan from a closed list of metrics, dimensions and filters. pandas runs the plan locally and the answer is written "
            "from the executed result, so every number in the answer comes from the data."),
    "trigger": ("Runs when an analyst submits a question. Questions about forecasts, causes, individual accounts, raw rows, credentials or "
                "anything outside the five tables are declined before any planner sees them."),
    "false_positive": ("A confident but wrong answer: the planner reads the question differently from the analyst (for example 'risky' taken as "
                       "default rate, or the wrong segment), or a question that should have been declined gets answered. The exact plan and "
                       "query are shown with every answer so this can be spotted."),
    "false_negative": ("A valid question that is declined or only partly understood (for example 'more than 2 missed payments', which is not a "
                       "supported filter), or the free-tier LLM being rate limited so the offline interpreter handles it. Cost: the analyst "
                       "rephrases or picks an example question."),
    "human": ("The analyst reads the answer and the 'Where this came from' panel (metric, filters, grouping, source) before using a figure. "
              "Nothing is written back to any system, and the LLM never sees row-level data or results."),
    "data_sent": ("Only the question text and the table catalog (metric and column descriptions plus category names) are sent to the LLM. "
                  "No rows, no computed results, no identifiers."),
}

ANOMALY_NOTES = {
    "title": "Anomaly detection",
    "how": ("Each account is compared with its peer segment (same loan type and income band) and with its own history (last 3 months versus "
            "first 9). A signal fires when a value is beyond the peer group's 99th percentile and above a materiality floor, or when the "
            "account's own payment behaviour breaks sharply. Accounts with two or more signals, or a sharp own-history break, are flagged."),
    "trigger": ("Runs on every data build over all accounts. An account is flagged when at least two independent signals fire, or when its "
                "days past due jump by 25+ days (to 30+) or its paid ratio drops by 50+ points between months 1-9 and months 10-12."),
    "false_positive": ("A flagged account that is actually fine: a one-off payment pause, a bank processing error, a customer with many "
                       "legitimate credit lines, or an unusual but healthy account in a small peer group. The analyst dismisses it with a note."),
    "false_negative": ("A risky account that is not flagged: slow deterioration that never crosses the 99th-percentile or two-signal bar, or an "
                       "account with no bureau file (bureau signals cannot fire). The model risk score is the second line of defence."),
    "human": ("The analyst opens each flagged account, reads the reasons with their numbers and records Confirm or Dismiss with a note. "
              "Each decision is stored with timestamp, account, decision, note and reviewer. Decisions never change the model or the account."),
}
