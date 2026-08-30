"""Microduck simulation core.

Runs the REAL policies Hugging Face / Pollen Robotics ship with Microduck. The
MJCFs and the ONNX checkpoints are fetched at runtime from the official
playground Space `pollen-robotics/microduck-simulator`, so nothing is vendored
here and there is no re-implementation of the robot.

Physics: MuJoCo, timestep 0.005 s, decimation 4 -> 50 Hz control.
The observation layout (61D), the action scaling, the command encodings and the
one-shot state machines are ported from that Space's `app/src/game/game.js` and
`constants.js`, which are the reference implementation:

    obs = [base_ang_vel(3), projected_gravity(3), joint_pos(14),
           joint_vel(14), last_action(14), command(13)]
    ctrl[j] = DEFAULT_POSE[j] + action[j] * ACTION_SCALE

The 13-slot command is overloaded per mode:
    walk / drive : cmd[0:3] = vx, vy, wz
    sitstand     : cmd[0]   = sit flag (0 = stand up, 1 = sit down)
    groundpick   : cmd[0:2] = cos, sin of a 4.0 s phase clock (exits at 0.7)
    crouch       : cmd[0:2] = cos, sin of a 5.0 s phase clock (exits at 0.7)
    all but those: cmd[3:7] = head [neck_pitch, head_pitch, head_yaw, head_roll],
                              EMA-smoothed at 50 Hz (alpha 0.2)
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

# A CPU Space has no GPU and no display, so MuJoCo's default EGL/GLFW backends
# cannot make a context. osmesa is the software rasteriser (libosmesa6 in
# packages.txt). Must be set before mujoco is imported.
if sys.platform.startswith("linux") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# -- Constants (from app/src/game/constants.js) --------------------------
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
DEFAULT_POSE = np.array([
    0, -0.08726646259971647, -0.457924, -0.004940, 0.452984,
    0.3490658503988659, 0.3490658503988659, 0, 0,
    0, 0.08726646259971647, 0.457924, 0.004940, -0.452984,
], dtype=np.float32)

NUM_JOINTS = 14
OBS_SIZE = 61
CMD_SIZE = 13
ACTION_SCALE = 1.0
TIMESTEP = 0.005
DECIMATION = 4
CTRL_DT = TIMESTEP * DECIMATION          # 50 Hz

# Legs velocity limits; then the roller branch (the real runtime launches
# rollers with --max-angular-vel 0.3 because faster turns tip it over).
VEL_FWD, VEL_BACK, VEL_ANG = 0.25, -0.2, 1.0
RVEL_FWD, RVEL_BACK, RVEL_ANG = 0.6, -0.5, 0.3

HEAD_MAX = 2.5           # rad at full deflection (runtime head_max)
HEAD_ALPHA = 0.2         # EMA per control step

BALL_RADIUS = 0.05
BALL_PARK = "8 0 0.05"   # off-camera, but not far enough to skew stat.extent
# The official playground boxes the duck into a 3 m arena because a player
# steers it and a ball has to stay in play. Here the routine is scripted and
# the camera tracks, so walls only ever produce crashes: skating sustains
# ~0.55 m/s and cleared the 1.5 m arena in under 4 s. The walls are kept as a
# far-out safety net (never reached inside the 30 s cap) and stay invisible.
ARENA_HALF, WALL_T, WALL_H = 6.0, 0.05, 0.25
SPAWN_X, SPAWN_Y = 0.0, 0.0

KICK_STEPS = 25
POST_KICK_LOCK_STEPS = 20
GROUND_PICK_PERIOD_S = 4.0
GROUND_PICK_END_PHASE = 0.7
CROUCH_PERIOD_S = 5.0
CROUCH_END_PHASE = 0.7
SIT_HANDOVER_S = 0.8     # hold the stand under sitstand before commanding sit
SIT_STANDUP_S = 2.0      # let sitstand stand back up before leaving the mode
FALL_DEBOUNCE_STEPS = 10
FALL_SETTLE_STEPS = 15
RECOVER_UPRIGHT_STEPS = 50
RECOVER_GIVEUP_STEPS = 300

POLICY_FILES = {
    "walk": "BEST_alpha_walking.onnx",
    "stand": "BEST_alpha_stand.onnx",
    "sitstand": "BEST_alpha_sitstand.onnx",
    "kickL": "ball_kick_left.onnx",
    "kickR": "ball_kick_right.onnx",
    "roll": "roulade.onnx",
    "groundpick": "alpha_ground_pick.onnx",
    "drive": "BEST_roller.onnx",
    "crouch": "BEST_roller_crouch.onnx",
}
LEGS_POLICIES = ("walk", "stand", "sitstand", "kickL", "kickR", "roll", "groundpick")
ROLLER_POLICIES = ("drive", "crouch")

VARIANTS = {
    "legs": {"xml": "robot_allcollisions.xml", "policies": LEGS_POLICIES},
    "rollers": {"xml": "robot_allcollisions_rollers.xml", "policies": ROLLER_POLICIES},
}

SPACE = ("https://huggingface.co/spaces/pollen-robotics/microduck-simulator"
         "/resolve/main/app/public")
CACHE = os.environ.get("MICRODUCK_CACHE") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".microduck_cache")


# -- Asset fetch ---------------------------------------------------------
def _get(rel, dest):
    """Download `rel` from the simulator Space into `dest` once."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(SPACE + "/" + rel, timeout=120) as r:
        blob = r.read()
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, dest)
    return dest


def ensure_assets(variant="legs"):
    """Fetch one variant's MJCF + meshes + policies.

    Returns (mjcf_path, mesh_dir, policy_dir). Meshes are shared between the
    two variants, so the roller switch only pulls the 5 extra wheel parts.
    """
    spec = VARIANTS[variant]
    mjcf = _get("robot/mjlab/" + spec["xml"], os.path.join(CACHE, spec["xml"]))
    mesh_dir = os.path.join(CACHE, "assets")
    root = ET.parse(mjcf).getroot()
    for m in root.find("asset").iter("mesh"):
        _get("robot/mjlab/meshes/" + m.get("file"),
             os.path.join(mesh_dir, m.get("file")))
    policy_dir = os.path.join(CACHE, "policies")
    for key in spec["policies"]:
        _get("policies/" + POLICY_FILES[key],
             os.path.join(policy_dir, POLICY_FILES[key]))
    return mjcf, mesh_dir, policy_dir


# -- Model assembly ------------------------------------------------------
def build_xml(mjcf_path, mesh_dir):
    """Robot MJCF + floor, fence, ball, lights and a STAND key.

    Mirrors buildPhysicsXml() in game.js, except the `visual` geoms are KEPT:
    they are contype=0/conaffinity=0 so they cost nothing physically, and they
    are what makes the render look like an actual duck.
    """
    root = ET.parse(mjcf_path).getroot()
    root.find("compiler").set("meshdir", mesh_dir)
    ET.SubElement(root, "option", {"timestep": str(TIMESTEP)})

    asset = root.find("asset")
    ET.SubElement(asset, "texture", {
        "name": "sky", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.30 0.50 0.82", "rgb2": "0.88 0.94 1.0",
        "width": "256", "height": "256"})
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "width": "512", "height": "512",
        "rgb1": "0.93 0.93 0.96", "rgb2": "0.78 0.81 0.88"})
    ET.SubElement(asset, "material", {
        "name": "gridmat", "texture": "grid", "texrepeat": "14 14",
        "reflectance": "0.08"})

    world = root.find("worldbody")
    ET.SubElement(world, "geom", {
        "name": "floor", "type": "plane", "size": "0 0 0.05",
        "pos": "0 0 0", "material": "gridmat"})
    ET.SubElement(world, "light", {
        "pos": "0.6 -0.6 1.8", "dir": "-0.3 0.3 -1",
        "directional": "true", "diffuse": "0.75 0.75 0.75"})
    ET.SubElement(world, "light", {
        "pos": "-1 1 1.2", "dir": "0.5 -0.5 -1",
        "directional": "true", "diffuse": "0.25 0.25 0.3"})

    off, span = ARENA_HALF + WALL_T, ARENA_HALF + 0.05
    for name, pos, size in [
        ("wall_px", str(off) + " 0 " + str(WALL_H), str(WALL_T) + " " + str(span) + " " + str(WALL_H)),
        ("wall_nx", str(-off) + " 0 " + str(WALL_H), str(WALL_T) + " " + str(span) + " " + str(WALL_H)),
        ("wall_py", "0 " + str(off) + " " + str(WALL_H), str(span) + " " + str(WALL_T) + " " + str(WALL_H)),
        ("wall_ny", "0 " + str(-off) + " " + str(WALL_H), str(span) + " " + str(WALL_T) + " " + str(WALL_H)),
    ]:
        # Invisible fence keeps the duck and ball in frame. group 3 = not drawn.
        ET.SubElement(world, "geom", {
            "name": name, "type": "box", "pos": pos, "size": size, "group": "3"})

    # Flat discs the learner walks to. Parked off-screen until a lesson moves
    # them; visual only (contype/conaffinity 0) so they never trip the duck.
    for i, rgba in enumerate(("0.85 0.25 0.15 0.9", "0.20 0.45 0.80 0.9")):
        ET.SubElement(world, "geom", {
            "name": "target_%d" % i, "type": "cylinder", "size": "0.075 0.002",
            "pos": "%d 0 0.002" % (20 + i), "rgba": rgba,
            "contype": "0", "conaffinity": "0", "group": "2"})

    ball = ET.SubElement(world, "body", {"name": "ball", "pos": BALL_PARK})
    ET.SubElement(ball, "freejoint", {"name": "ball_freejoint"})
    ET.SubElement(ball, "geom", {
        "name": "ball_geom", "type": "sphere", "size": str(BALL_RADIUS),
        "mass": "0.03", "friction": "0.4 0.01 0.003", "solref": "0.03 0.4",
        "condim": "6", "rgba": "0.95 0.35 0.15 1"})

    # STAND keyframe: freejoint(7) + every hinge in document order + ball(7).
    # The roller variant has 18 hinges (4 passive wheels), which default to 0.
    pose = dict(zip(JOINT_NAMES, DEFAULT_POSE))
    order = [j.get("name") for body in root.iter("body")
             for j in body if j.tag == "joint"]
    qj = " ".join(str(float(pose.get(n, 0.0))) for n in order)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", {
        "name": "STAND",
        "qpos": (str(SPAWN_X) + " " + str(SPAWN_Y) + " 0.12 1 0 0 0 "
                 + qj + " " + BALL_PARK + " 1 0 0 0"),
        "ctrl": " ".join(str(float(x)) for x in DEFAULT_POSE)})
    return ET.tostring(root, encoding="unicode")


# -- The sim -------------------------------------------------------------
class Microduck:
    def __init__(self, width=640, height=360, render=True, variant="legs"):
        import mujoco
        import onnxruntime as ort
        self.mj = mujoco
        self.variant = variant
        mjcf, mesh_dir, policy_dir = ensure_assets(variant)
        self.model = mujoco.MjModel.from_xml_string(build_xml(mjcf, mesh_dir))
        self.data = mujoco.MjData(self.model)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sessions = {
            k: ort.InferenceSession(
                os.path.join(policy_dir, POLICY_FILES[k]), sess_options=opts,
                providers=["CPUExecutionProvider"])
            for k in VARIANTS[variant]["policies"]}

        self.qadr = [self.model.jnt(n).qposadr[0] for n in JOINT_NAMES]
        self.dadr = [self.model.jnt(n).dofadr[0] for n in JOINT_NAMES]
        self.gyro = self.model.sensor("imu_ang_vel").adr[0]
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.stand_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
        self.markers = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                          "target_%d" % i) for i in range(2)]
        self.ball_q = self.model.jnt("ball_freejoint").qposadr[0]
        self.ball_d = self.model.jnt("ball_freejoint").dofadr[0]

        # Shadow mapping costs ~90 ms/frame here and only buys a blocky
        # contact shadow, so it is off; multisampling is cheap and kept.
        self.model.vis.quality.shadowsize = 0
        self.model.vis.quality.offsamples = 4

        # MuJoCo scales the near/far planes by model.stat.extent, which it
        # derives from the bounding box - and the ball parked far off-screen
        # blew that up to ~52 m, putting the near plane at 0.52 m. The chase
        # camera sits 0.62 m out, so the near plane sliced through the duck
        # and clipped the neck away: the head rendered as if detached.
        # Pin the extent to the robot's own scale instead.
        self.model.stat.extent = 0.6
        self.model.vis.map.znear = 0.01     # 6 mm
        self.model.vis.map.zfar = 300.0     # 180 m, keeps the horizon

        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.opt = mujoco.MjvOption()
        for g in range(len(self.opt.geomgroup)):
            self.opt.geomgroup[g] = 1 if g <= 2 else 0   # hide collision + fence
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = self.trunk
        self.cam.distance = 1.25
        self.cam.elevation = -22.0
        self.cam.lookat[:] = [0.0, 0.0, 0.05]
        self.cam_offset = 125.0     # degrees behind-left of the duck's heading
        self._az = None
        self.reset()

    def set_targets(self, points):
        """Place the goal discs. Anything not given is parked out of frame."""
        for i, gid in enumerate(self.markers):
            if i < len(points):
                self.model.geom_pos[gid] = [points[i][0], points[i][1], 0.002]
            else:
                self.model.geom_pos[gid] = [20 + i, 0, 0.002]
        self.mj.mj_forward(self.model, self.data)

    def vel_limits(self):
        """(forward, backward, yaw) for this locomotion variant."""
        if self.variant == "rollers":
            return RVEL_FWD, RVEL_BACK, RVEL_ANG
        return VEL_FWD, VEL_BACK, VEL_ANG

    # -- state ------------------------------------------------------------
    def reset(self):
        self.mj.mj_resetDataKeyframe(self.model, self.data, self.stand_key)
        self.mj.mj_forward(self.model, self.data)
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
        self.head_target = np.zeros(4, dtype=np.float32)
        self.head_smooth = np.zeros(4, dtype=np.float32)
        self.recovery = None
        self.fall_debounce = 0
        self.post_kick_lock = 0
        self.ball_active = False

    def proj_gravity(self):
        w, x, y, z = self.data.body(self.trunk).xquat
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        return R.T @ np.array([0.0, 0.0, -1.0])

    def yaw(self):
        q = self.data.qpos
        return float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]),
                                1 - 2 * (q[5] ** 2 + q[6] ** 2)))

    # Where the ball has to sit for the kick to actually connect, in the
    # duck's own frame. The kick policies are BLIND (no ball in the 61-D obs)
    # -- on the real robot the operator aims the duck first. A scripted
    # routine has no operator, so the ball is placed on the swinging foot's
    # path instead. Swept empirically: 0.12 m ahead and 0.06 m to the kicking
    # side moves the ball 0.25-0.38 m and leaves the duck upright; further out
    # (the 0.30-0.35 m the interactive playground uses) simply misses.
    KICK_BALL_FWD = 0.12
    KICK_BALL_LAT = 0.06

    def spawn_ball(self, rng, foot="R"):
        q, v = self.data.qpos, self.data.qvel
        yaw = self.yaw()
        fwd = self.KICK_BALL_FWD + (rng.random() - 0.5) * 0.02
        lat = (self.KICK_BALL_LAT + (rng.random() - 0.5) * 0.02) * (1 if foot == "L" else -1)
        margin = BALL_RADIUS + 0.05
        lo, hi = -ARENA_HALF + margin, ARENA_HALF - margin
        bx = q[0] + np.cos(yaw) * fwd - np.sin(yaw) * lat
        by = q[1] + np.sin(yaw) * fwd + np.cos(yaw) * lat
        q[self.ball_q] = float(np.clip(bx, lo, hi))
        q[self.ball_q + 1] = float(np.clip(by, lo, hi))
        q[self.ball_q + 2] = BALL_RADIUS + 0.005
        q[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
        v[self.ball_d:self.ball_d + 6] = 0
        self.mj.mj_forward(self.model, self.data)
        self.ball_active = True

    def park_ball(self):
        q, v = self.data.qpos, self.data.qvel
        q[self.ball_q:self.ball_q + 3] = [50, 0, BALL_RADIUS]
        q[self.ball_q + 3:self.ball_q + 7] = [1, 0, 0, 0]
        v[self.ball_d:self.ball_d + 6] = 0
        self.mj.mj_forward(self.model, self.data)
        self.ball_active = False

    # -- one control step (50 Hz) -----------------------------------------
    def control_step(self, mode, cmd3, phase=None, sit_flag=0.0):
        """Advance 1/50 s. `mode` is a policy key; `phase` drives the
        groundpick/crouch clocks; `sit_flag` is the sitstand posture bit."""
        if self.recovery is not None:
            policy = "stand" if self.recovery["state"] == "recovering" else None
        elif self.variant == "rollers" and mode == "walk":
            policy = "drive"          # skating replaces the walker on rollers
        else:
            policy = mode

        if policy is not None:
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            i = 0
            obs[i:i + 3] = self.data.sensordata[self.gyro:self.gyro + 3]
            i += 3
            obs[i:i + 3] = self.proj_gravity()
            i += 3
            obs[i:i + NUM_JOINTS] = self.data.qpos[self.qadr] - DEFAULT_POSE
            i += NUM_JOINTS
            obs[i:i + NUM_JOINTS] = self.data.qvel[self.dadr]
            i += NUM_JOINTS
            obs[i:i + NUM_JOINTS] = self.last_action
            i += NUM_JOINTS

            cmd = np.zeros(CMD_SIZE, dtype=np.float32)
            if mode == "sitstand":
                cmd[0] = sit_flag
            elif mode in ("groundpick", "crouch") and phase is not None:
                a = 2 * np.pi * phase
                cmd[0], cmd[1] = np.cos(a), np.sin(a)
            elif self.recovery is None and self.post_kick_lock == 0:
                cmd[:3] = cmd3

            # Head slots, EMA-smoothed once per control step like the runtime.
            self.head_smooth += HEAD_ALPHA * (self.head_target - self.head_smooth)
            # The runtime zero-pads the head slots for the pick policy and for
            # fall recovery (mjlab's zero_command_padding).
            if mode != "groundpick" and self.recovery is None:
                cmd[3:7] = self.head_smooth
            obs[i:i + CMD_SIZE] = cmd

            act = self.sessions[policy].run(None, {"obs": obs.reshape(1, -1)})[0][0]
            self.last_action = act.astype(np.float32)
            self.data.ctrl[:NUM_JOINTS] = DEFAULT_POSE + act * ACTION_SCALE

        for _ in range(DECIMATION):
            self.mj.mj_step(self.model, self.data)

        if self.post_kick_lock > 0:
            self.post_kick_lock -= 1
        self._update_recovery(mode)

    def _update_recovery(self, mode):
        z, gz = float(self.data.qpos[2]), float(self.proj_gravity()[2])
        if not np.isfinite(z) or not np.isfinite(gz):
            self.reset()
            return
        fallen = gz > -0.5 or z < 0.02
        if self.recovery is not None:
            self.recovery["steps"] += 1
            if self.recovery["state"] == "fallen":
                if self.recovery["steps"] >= FALL_SETTLE_STEPS:
                    self.recovery = {"state": "recovering", "steps": 0, "upright": 0}
                    self.last_action[:] = 0
            else:
                self.recovery["upright"] = (
                    self.recovery["upright"] + 1 if gz < -0.85 else 0)
                if self.recovery["upright"] >= RECOVER_UPRIGHT_STEPS:
                    self.recovery = None
                    self.last_action[:] = 0
                elif self.recovery["steps"] >= RECOVER_GIVEUP_STEPS:
                    self.reset()
        # The get-up policy is trained on the legs model in the walk mode only
        # (game.js: recoverable = loco === "legs" && mode === "walk").
        elif (fallen and self.variant == "legs" and mode == "walk"
                and self.post_kick_lock == 0):
            self.fall_debounce += 1
            if self.fall_debounce >= FALL_DEBOUNCE_STEPS:
                self.fall_debounce = 0
                self.recovery = {"state": "fallen", "steps": 0}
        else:
            self.fall_debounce = 0

    def frame(self):
        """Render one frame from a tracking camera that eases in behind the duck."""
        target = np.degrees(self.yaw()) + self.cam_offset
        if self._az is None:
            self._az = target
        else:
            # Shortest-arc smoothing, so a turn pans instead of snapping.
            self._az += 0.05 * (((target - self._az + 180.0) % 360.0) - 180.0)
        self.cam.azimuth = self._az
        self.renderer.update_scene(self.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()
