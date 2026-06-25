"""Scripted closed-loop oracle that actually solves NutAssembly.

This is an EXPERT, not the vision deliverable: it reads privileged simulator state
(nut handle site, nut center, peg position, end-effector pose) to drive a finite
state machine via the OSC_POSE controller. Its purpose is twofold:
  1. a baseline that demonstrably completes the task, and
  2. the demonstration generator for behaviour cloning (see src/demos/collect_demos.py).

Control law: OSC_POSE maps action[0:3] -> position delta (x0.05 m/step),
action[3:6] -> orientation delta, action[6] -> gripper (+1 close / -1 open).
We servo with a proportional law `action = clip((target - current)/0.05, -1, 1)`.

Key geometric insight: the gripper grasps the nut by its protruding HANDLE, which is
offset from the nut's center/hole. During transport we therefore servo the NUT
CENTER (not the end-effector) onto the peg, and we align the finger-closing axis
perpendicular to the handle rod so the fingers actually pinch it.
"""
import numpy as np

from ..base_controller import BaseController


def _wrap(a):
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class OracleController(BaseController):
    def __init__(self, env, lift_height=0.30, insert_height=0.11, place_tol=0.004):
        super().__init__()
        self.env = env
        self.lift_height = lift_height
        self.insert_height = insert_height
        self.place_tol = place_tol

    def reset(self):
        pass

    # --- privileged-state accessors (read live; sim is rebuilt on env.reset) ---
    @property
    def _sim(self):
        return self.env.sim

    @property
    def _arm(self):
        r = self.env.robots[0]
        sid = r.eef_site_id
        return r, (sid[r.arms[0]] if isinstance(sid, dict) else sid)

    def _eef(self):
        _, sid = self._arm
        return np.asarray(self._sim.data.site_xpos[sid])

    def _geom(self, name):
        return np.asarray(self._sim.data.geom_xpos[self._sim.model.geom_name2id(name)])

    def _closing_axis_angle(self):
        """World-frame angle of the line joining the two finger pads."""
        g = self.env.robots[0].gripper["right"]
        lp = np.mean([self._geom(x) for x in g.important_geoms["left_fingerpad"]], axis=0)
        rp = np.mean([self._geom(x) for x in g.important_geoms["right_fingerpad"]], axis=0)
        d = rp - lp
        return np.arctan2(d[1], d[0])

    def _handle(self, nut):
        name = nut.important_sites["handle"]
        return np.asarray(self._sim.data.site_xpos[self._sim.model.site_name2id(name)])

    def _nut_center(self, nut):
        return np.asarray(self._sim.data.body_xpos[self.env.obj_body_id[nut.name]])

    def _peg(self, nut_idx):
        bid = self.env.peg1_body_id if nut_idx == 0 else self.env.peg2_body_id
        return np.asarray(self._sim.data.body_xpos[bid]).copy()

    def _table_z(self):
        return self.env.table_offset[2]

    # --- low-level command ---
    def _step(self, target_xyz, grip, K, target_yaw, align=True):
        eef = self._eef()
        a = np.zeros(7)
        a[0:3] = np.clip(K * (np.asarray(target_xyz) - eef) / 0.05, -1.0, 1.0)
        a[6] = grip
        if align:
            yaw_err = _wrap(target_yaw - self._closing_axis_angle())
            a[5] = np.clip(2.5 * yaw_err / 0.5, -1.0, 1.0)
        self._last_action = a
        return a, self.env.step(a)

    def place_nut(self, nut_idx, on_step=None):
        """Run the full pick-place-insert FSM for one nut.

        Args:
            nut_idx: 0 == square nut (peg1), 1 == round nut (peg2).
            on_step: optional callback(action, obs, reward, done, info) invoked every
                step (used by demo collection to record image/action pairs).
        Returns:
            True if the env reports the task solved.
        """
        nut = self.env.nuts[nut_idx]
        peg = self._peg(nut_idx)
        tz = self._table_z()

        # finger-closing axis perpendicular to the handle rod (rod = center -> handle)
        rod = self._handle(nut)[:2] - self._nut_center(nut)[:2]
        target_yaw = _wrap(np.arctan2(rod[1], rod[0]) + np.pi / 2)

        last_done = False

        def run(target_fn, grip, K, max_steps, stop_fn=None, align=True):
            nonlocal last_done
            for _ in range(max_steps):
                tgt = target_fn()
                action, (obs, reward, done, info) = self._step(tgt, grip, K, target_yaw, align=align)
                last_done = done
                if on_step is not None:
                    on_step(action, obs, reward, done, info)
                if done:
                    return
                if stop_fn is not None and stop_fn():
                    return

        H = lambda: self._handle(nut)
        N = lambda: self._nut_center(nut)
        E = self._eef

        # 1. hover above the handle (gripper open), aligning yaw
        run(lambda: [*H()[:2], H()[2] + 0.10], -1.0, 1.2, 150,
            stop_fn=lambda: np.linalg.norm(E()[:2] - H()[:2]) < 0.005)
        # 2. descend onto the handle
        run(lambda: [*H()[:2], H()[2]], -1.0, 1.2, 120,
            stop_fn=lambda: abs(E()[2] - H()[2]) < 0.004)
        # 3. close the gripper (no xy push while clamping)
        run(lambda: [*H()[:2], H()[2]], 1.0, 0.0, 25)
        # 4. lift gently
        run(lambda: [*H()[:2], tz + self.lift_height], 1.0, 0.8, 120,
            stop_fn=lambda: E()[2] > tz + self.lift_height - 0.02)
        # 5. transport: servo the NUT CENTER over the peg
        run(lambda: [E()[0] + (peg[0] - N()[0]), E()[1] + (peg[1] - N()[1]), tz + self.lift_height],
            1.0, 1.0, 200, stop_fn=lambda: np.linalg.norm(peg[:2] - N()[:2]) < self.place_tol)
        # 6. lower the nut onto the peg
        run(lambda: [E()[0] + (peg[0] - N()[0]), E()[1] + (peg[1] - N()[1]), tz + self.insert_height],
            1.0, 0.7, 160)
        # 7. release
        run(lambda: [*N()[:2], N()[2]], -1.0, 0.0, 15)
        # 8. retreat
        run(lambda: [E()[0], E()[1], tz + self.lift_height], -1.0, 0.6, 60)

        return bool(self.env._check_success())

    # `act` keeps the BaseController interface; the FSM lives in place_nut/_step.
    _last_action = None

    def act(self, obs):
        raise NotImplementedError(
            "OracleController drives the env directly via place_nut(); it is an expert "
            "demo generator, not a per-step act() policy."
        )
