import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from packing_env import BinPackingEnv
from train_model import MLPPolicy
import torch
import argparse
import os

def _score_positions(height_map: np.ndarray, w: int, h: int, canvas_width: int):
    scores = []
    current_max = float(np.max(height_map)) if height_map.size else 0.0
    for x in range(0, canvas_width - w + 1):
        region = height_map[x : x + w]
        y = float(np.max(region)) if region.size else 0.0
        new_local = y + h
        new_max = max(current_max, new_local)
        max_increase = new_max - current_max
        
        height_score = -y * 2.0
        left_h = height_map[x - 1] if x - 1 >= 0 else y
        right_h = height_map[x + w] if x + w < canvas_width else y
        
        hole_fill = max(0.0, min(left_h, right_h) - y)
        hole_score = hole_fill * 3.0
        pillar_penalty = max_increase * 2.0
        
        local_mean = float(np.mean(region)) if region.size else y
        smooth_penalty = abs(new_local - local_mean) * 0.2
        
        score = height_score + hole_score - pillar_penalty - smooth_penalty
        scores.append((score, x, int(y)))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlpModelPath", type=str, default="PackingModelLast.pth")
    parser.add_argument("--topK", type=int, default=5)
    parser.add_argument("--lambdaNN", type=float, default=0.7)
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()


    env = BinPackingEnv(canvas_width=args.width, num_rects=args.num)
    obs, _ = env.reset(seed=args.seed)


    if not os.path.exists(args.mlpModelPath):
        raise FileNotFoundError(f"MLP model not found: {args.mlpModelPath}")
        
    model = MLPPolicy(input_dim=env.observation_space.shape[0])
    state = torch.load(args.mlpModelPath, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    done = False
    while not done:
        w, h = env.rects[env.current_idx]
        

        scored = _score_positions(env.height_map, w, h, env.canvas_width)
        scored.sort(key=lambda t: t[0], reverse=True)
        candidates = scored[: max(1, args.topK)]
        
        with torch.no_grad():
            x_pred = model(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).item()
        
        max_x = env.canvas_width - w
        combined = []
        for s, x, y in candidates:
            x_norm = (x / max_x) if max_x > 0 else 0.0

            nn_score = -abs(x_norm - x_pred)

            combo = args.lambdaNN * nn_score + (1.0 - args.lambdaNN) * s
            combined.append((combo, x))

        combined.sort(key=lambda t: t[0], reverse=True)
        best_x = combined[0][1]
        
        action = np.array([(best_x / max_x) if max_x > 0 else 0.0], dtype=np.float32)
        obs, reward, done, _, info = env.step(action)


    metrics = env.get_metrics()
    print("\nResults:")
    print(f"Efficiency: {metrics['efficiency']:.2%}")
    print(f"Area Utilization: {metrics['area_utilization']:.2%}")
    print(f"Max Height: {metrics['max_height']:.1f}")
    print(f"Ideal Height: {metrics['ideal_height']:.1f}")


    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, args.width)
    ax.set_ylim(0, max(metrics["max_height"], args.width) * 1.1)
    
    for x, y, w, h in env.placed_rects:
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1,
            edgecolor="black",
            facecolor=np.random.rand(3),
        )
        ax.add_patch(rect)
        
    ax.axhline(y=metrics["ideal_height"], color="r", linestyle="--", label="Ideal Height")
    ax.axhline(y=metrics["max_height"], color="g", linestyle="-", label="Actual Height")
    
    plt.title(f"Neural-Guided Packing\nEfficiency: {metrics['efficiency']:.2%}")
    plt.legend()
    plt.show()