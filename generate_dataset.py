import numpy as np
import os
from packing_env import BinPackingEnv
from heuristic_solver import SkylineHeuristic, ShelfHeuristic


def generate(
    num_episodes=10000,
    canvas_width=100,
    num_rects=50,
    seed=1337,
    out_path="dataset_binpacking.npz",
    teacher="shelf",
):
    rng = np.random.default_rng(seed)
    solver = (
        ShelfHeuristic(canvas_width=canvas_width)
        if teacher == "shelf"
        else SkylineHeuristic(canvas_width=canvas_width)
    )

    obs_list = []
    act_list = []

    for ep in range(num_episodes):
        env_seed = int(rng.integers(0, 1_000_000))
        env = BinPackingEnv(canvas_width=canvas_width, num_rects=num_rects)
        obs, _ = env.reset(seed=env_seed)

        done = False
        while not done:
            idx = env.current_idx
            if idx >= env.num_rects:
                break
            w, h = env.rects[idx]

            if teacher == "shelf":
                x, y, _ = SkylineHeuristic(
                    canvas_width=canvas_width
                ).find_best_position(env.height_map, w, h)
            else:
                x, y, _ = solver.find_best_position(env.height_map, w, h)
            max_x = env.canvas_width - w
            x_norm = (x / max_x) if max_x > 0 else 0.0

            obs_list.append(obs.copy())
            act_list.append([x_norm])

            action = np.array([x_norm], dtype=np.float32)
            obs, _, done, _, _ = env.step(action)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep + 1}/{num_episodes} completed")

    obs_arr = np.array(obs_list, dtype=np.float32)
    act_arr = np.array(act_list, dtype=np.float32)

    np.savez_compressed(out_path, observations=obs_arr, actions=act_arr)
    print(f"Saved dataset: {out_path}")


if __name__ == "__main__":
    generate(teacher="shelf")
