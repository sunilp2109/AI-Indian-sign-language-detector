"""Shared landmark feature layout. Used by dataset scripts and later inference."""

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3
VALUES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
NUM_HANDS = 2
PRESENCE_FLAGS = 2

# left_xyz (63) + right_xyz (63) + left_present + right_present
FEATURE_DIM = VALUES_PER_HAND * NUM_HANDS + PRESENCE_FLAGS  # 128

WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
SCALE_EPSILON = 1e-6
