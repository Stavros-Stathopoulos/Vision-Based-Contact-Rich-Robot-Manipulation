"""Single source of truth for the NutAssembly vision pipeline.

Training and evaluation MUST share these values, otherwise the policy is fed
observations that differ from what it was trained on. Import from here instead
of re-declaring `suite.make(...)` arguments in every script.
"""

# --- Task / environment -----------------------------------------------------
# "NutAssemblySquare" is the single-nut curriculum (one nut + one peg) and is
# far more learnable than the full two-nut "NutAssembly". Switch to
# "NutAssembly" here once the square task is solved; everything else adapts.
TASK_NAME = "NutAssemblySquare"
ROBOT = "Panda"
GRIPPER = "PandaGripper"
CONTROLLER = "BASIC"
CAMERA_NAME = "agentview"
IMG_SIZE = 84
CONTROL_FREQ = 20
HORIZON = 200
REWARD_SHAPING = True

# Curriculum: shrinks the nut spawn region (and yaw range) around its default
# center so a successful grasp is repeatable enough to learn. 1.0 = full robosuite
# randomization (final target); smaller = easier. Raise toward 1.0 as success
# climbs. This changes task DIFFICULTY only — it does not reveal the nut's pose to
# the policy (still inferred from pixels).
RANDOMIZATION_SCALE = 0.25

# --- Observation ------------------------------------------------------------
# The ONLY two observation keys the policy is allowed to see. The nut's pose is
# deliberately absent: it must be inferred from `agentview_image`.
IMAGE_KEY = "agentview_image"
PROPRIO_KEY = "robot0_proprio-state"  # robot joints/eef/gripper only — NO object state
NUM_STACK = 3  # stacked camera frames -> gives the CNN temporal (velocity) cues

# --- Artifacts --------------------------------------------------------------
LOG_DIR = "./rl_logs"
MODEL_PATH = "sac_nut_assembly"        # final model  (SB3 appends .zip)
BEST_MODEL_PATH = "sac_nut_assembly_best"  # best-by-true-success checkpoint
