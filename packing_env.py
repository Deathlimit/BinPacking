import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BinPackingEnv(gym.Env):
    def __init__(
        self,
        canvas_width=100,
        num_rects=50,
        min_w=5,
        max_w=20,
        min_h=5,
        max_h=20,
        reward_cfg=None,
    ):
        super(BinPackingEnv, self).__init__()

        self.canvas_width = canvas_width
        self.num_rects = num_rects
        self.min_w = min_w
        self.max_w = max_w
        self.min_h = min_h
        self.max_h = max_h

        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)


        self.obs_len = self.canvas_width + 4
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.obs_len,), dtype=np.float32
        )


        self.reward_cfg = reward_cfg or {
            "base": 0.1,
            "height_increase_penalty": 0.05,
            "final_eff_bonus": 12.0,
            "spike_penalty_coef": 0.1,
        }

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)


        self.rects = []
        for _ in range(self.num_rects):
            w = np.random.randint(self.min_w, self.max_w + 1)
            h = np.random.randint(self.min_h, self.max_h + 1)
            self.rects.append((w, h))


        self.total_area = sum(w * h for w, h in self.rects)
        self.ideal_height = self.total_area / self.canvas_width


        self.height_map = np.zeros(self.canvas_width, dtype=np.float32)
        self.current_idx = 0
        self.placed_rects = []
        self.max_height = 0

        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_idx >= self.num_rects:
            next_w, next_h = 0, 0
        else:
            next_w, next_h = self.rects[self.current_idx]


        scale_factor = max(self.max_height, 1.0)
        normalized_map = self.height_map / scale_factor


        efficiency = 0.0
        if self.max_height > 0:
            efficiency = self.ideal_height / self.max_height

        obs = np.concatenate(
            [
                normalized_map,
                [
                    next_w / self.max_w,
                    next_h / self.max_h,
                    efficiency,
                    self.current_idx / self.num_rects,
                ],
            ]
        ).astype(np.float32)

        return obs

    def step(self, action):
        if self.current_idx >= self.num_rects:
            return self._get_obs(), 0, True, False, {}


        if isinstance(action, (list, tuple, np.ndarray)):
            x_norm = float(action[0])
        else:
            x_norm = float(action)
        w, h = self.rects[self.current_idx]

        max_x = self.canvas_width - w
        x = int(x_norm * max_x)
        x = max(0, min(x, max_x))


        region = self.height_map[x : x + w]
        y = np.max(region)


        new_height = y + h
        self.height_map[x : x + w] = new_height
        self.placed_rects.append((x, y, w, h))

        prev_max_height = self.max_height
        self.max_height = np.max(self.height_map)

        eff_raw = self.ideal_height / self.max_height if self.max_height > 0 else 0.0

        placed_area = sum(wi * hi for _, _, wi, hi in self.placed_rects)
        bounding_area = float(self.canvas_width * max(self.max_height, 1))
        area_util = (placed_area / bounding_area) if bounding_area > 0 else 0.0

        eff_adjusted = (0.7 * eff_raw + 0.3 * area_util) ** 0.5

        height_increase = self.max_height - prev_max_height


        reward = float(self.reward_cfg.get("base", 0.1))


        reward -= height_increase * float(
            self.reward_cfg.get("height_increase_penalty", 0.05)
        )

        left_x = max(0, x - 1)
        right_x = min(self.canvas_width - 1, x + w)
        left_h = float(self.height_map[left_x])
        right_h = float(self.height_map[right_x])
        local_spike = max(0.0, (new_height - max(left_h, right_h)))
        reward -= local_spike * float(self.reward_cfg.get("spike_penalty_coef", 0.1))

        self.current_idx += 1
        terminated = self.current_idx >= self.num_rects

        if terminated:
            final_efficiency = self.ideal_height / self.max_height
            reward += final_efficiency * float(
                self.reward_cfg.get("final_eff_bonus", 12.0)
            )

        return (
            self._get_obs(),
            reward,
            terminated,
            False,
            {
                "efficiency": eff_adjusted,
                "efficiency_raw": eff_raw,
                "area_utilization": area_util,
            },
        )

    def get_metrics(self):
        """Return detailed metrics including adjusted efficiency."""
        placed_area = sum(w * h for _, _, w, h in self.placed_rects)
        bounding_area = self.canvas_width * max(self.max_height, 1)
        eff_raw = self.ideal_height / self.max_height if self.max_height > 0 else 0.0
        area_util = (placed_area / bounding_area) if bounding_area > 0 else 0.0
        eff_adjusted = (0.7 * eff_raw + 0.3 * area_util) ** 0.5
        return {
            "efficiency": eff_adjusted,
            "efficiency_raw": eff_raw,
            "area_utilization": area_util,
            "placed_area": placed_area,
            "bounding_area": bounding_area,
            "ideal_height": self.ideal_height,
            "max_height": float(self.max_height),
        }

    def render(self):
        print(
            f"Placed: {len(self.placed_rects)}/{self.num_rects}, Max Height: {self.max_height}, Ideal: {self.ideal_height:.2f}"
        )
