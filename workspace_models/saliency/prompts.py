"""VLM prompts for saliency labeling (Workspace Models, Appendix C).

Verbatim templates. `{...}` fields are filled per task/frame. The pointing prompt follows
MolmoPoint's convention (Fig. 3 shows "Point at the <object>"). The event-detection prompts are
included for completeness (the long-horizon / keyframe path); the current-frame pipeline uses
only POINTING.
"""

# --- MolmoPoint: point at a persistent/salient object in the CURRENT frame (Fig. 3) ---
POINT_PROMPT = "Point at the {object}."

# --- Qwen3-VL: generate a per-task yes/no detector question (Appendix C) ---
EVENT_QUESTION_GEN_PROMPT = (
    "A robot is performing the task {task_description}. You will help detect important events "
    "that should be remembered from this demonstration in order for it to execute with those "
    "memories. Look across the subsampled frames and propose ONE visual yes/no question that can "
    "later be asked on a single frame at a time to decide whether that frame counts as a relevant "
    "event. The question should target a task-relevant event that would help capture a "
    "non-Markovian state change or a key subtask transition. It should be specific, visually "
    "answerable from one frame, and worded like a yes/no detector question. Example: \"Is the mug "
    "being picked up by the robot?\" Return ONLY a JSON object in exactly this format: "
    "{{\"prompt\": \"Is the mug being picked up by the robot?\", \"positive_description\": \"mug "
    "being picked up by the robot\"}} Do not include any other text."
)

# --- Qwen3-VL: classify a single frame against a detector question (Appendix C) ---
EVENT_DETECT_PROMPT = (
    "This is frame {frame_index} from a robot demonstration of the task {task_description} and the "
    "detection question is: {event_prompt}. Look ONLY at this single frame. Respond conservatively: "
    "mark the frame positive only if the event is visually happening in this frame. Return ONLY a "
    "JSON object in exactly this format: {{\"event\": true, \"reason\": \"short phrase\", "
    "\"probability\": 0.5}} Use false when the event is not happening. Do not include any other text."
)
