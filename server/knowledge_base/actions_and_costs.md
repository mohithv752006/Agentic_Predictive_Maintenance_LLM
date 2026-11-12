# Recommended actions and the cost model

Sources:
[1] This system's action and cost table, server/configs/actions.yaml.
[2] Backblaze Drive Stats data page (no official cost matrix is published). https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data

## Actions
monitor keeps the drive in service; schedule_replacement swaps it in the next maintenance window (about a week); replace_now swaps it immediately at a small urgency premium; escalate_to_human sends the case to an operator [1].

## Expected cost
Costs are in units of one planned replacement, and a missed failure costs cost_ratio units. The expected cost of an action is its fixed cost plus the probability of failing within its exposure horizon times cost_ratio: monitor is exposed for 30 days, schedule_replacement for 7 days, and replace_now has no exposure [1]. The action with the lowest expected cost is recommended. Because no official cost matrix exists, experiments sweep cost ratios of 10, 50 and 200 [1][2].

## Escalation
A case goes to a human when the fused prediction is too uncertain, when the specialist agents disagree strongly, or when feature drift for the drive model is at alert level. Escalation is never chosen on cost alone [1].
