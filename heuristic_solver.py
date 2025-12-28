import numpy as np
from typing import List, Tuple, Dict


class SkylineHeuristic:
    def __init__(self, canvas_width: int = 100):
        self.canvas_width = canvas_width

    def find_best_position(
        self, height_map: np.ndarray, w: int, h: int
    ) -> Tuple[int, int, float]:
        best_x, best_y = 0, 0
        best_score = -1e18

        current_max = float(np.max(height_map)) if height_map.size else 0.0

        for x in range(0, self.canvas_width - w + 1):
            region = height_map[x : x + w]
            y = float(np.max(region)) if region.size else 0.0


            new_local = y + h
            new_max = max(current_max, new_local)
            max_increase = new_max - current_max

            height_score = -y * 2.0

            left_h = height_map[x - 1] if x - 1 >= 0 else y
            right_h = height_map[x + w] if x + w < self.canvas_width else y
            hole_fill = max(0.0, min(left_h, right_h) - y)
            hole_score = hole_fill * 3.0


            pillar_penalty = max_increase * 2.0

            local_mean = float(np.mean(region)) if region.size else y
            smooth_penalty = abs(new_local - local_mean) * 0.2

            score = height_score + hole_score - pillar_penalty - smooth_penalty

            if score > best_score:
                best_score = score
                best_x, best_y = x, int(y)

        return best_x, best_y, best_score

    def solve(self, rects: List[Tuple[int, int]]) -> Dict:
        rects = list(rects)
        rects.sort(key=lambda r: r[0] * r[1], reverse=True)

        height_map = np.zeros(self.canvas_width, dtype=np.float32)
        placed: List[Tuple[int, int, int, int]] = []
        max_height = 0

        for w, h in rects:
            x, y, _ = self.find_best_position(height_map, w, h)
            new_h = y + h
            height_map[x : x + w] = new_h
            placed.append((x, y, w, h))
            if new_h > max_height:
                max_height = int(new_h)

        total_area = sum(w * h for w, h in rects)
        ideal_height = total_area / self.canvas_width
        efficiency = (ideal_height / max_height) if max_height > 0 else 0.0
        bounding_area = self.canvas_width * max_height
        area_utilization = (total_area / bounding_area) if bounding_area > 0 else 0.0

        return {
            "placed_rects": placed,
            "height_map": height_map,
            "max_height": max_height,
            "ideal_height": ideal_height,
            "efficiency": min(1.0, efficiency),
            "area_utilization": area_utilization,
            "total_area": total_area,
            "bounding_area": bounding_area,
        }

    def solve_beam(
        self,
        rects: List[Tuple[int, int]],
        depth: int = 2,
        top_k: int = 3,
        beam_size: int = 3,
    ) -> Dict:
        rects = list(rects)
        rects.sort(key=lambda r: r[0] * r[1], reverse=True)

        class State:
            __slots__ = ("height_map", "placed", "score", "max_height")

            def __init__(self, width):
                self.height_map = np.zeros(width, dtype=np.float32)
                self.placed = []
                self.score = 0.0
                self.max_height = 0.0

        beam: List[State] = [State(self.canvas_width)]

        for w, h in rects:
            candidates: List[State] = []
            for st in beam:
                scored: List[Tuple[float, int, int]] = []
                for x in range(0, self.canvas_width - w + 1):
                    region = st.height_map[x : x + w]
                    y = float(np.max(region)) if region.size else 0.0
                    current_max = (
                        float(np.max(st.height_map)) if st.height_map.size else 0.0
                    )
                    new_local = y + h
                    new_max = max(current_max, new_local)
                    max_increase = new_max - current_max
                    height_score = -y * 2.0
                    left_h = st.height_map[x - 1] if x - 1 >= 0 else y
                    right_h = st.height_map[x + w] if x + w < self.canvas_width else y
                    hole_fill = max(0.0, min(left_h, right_h) - y)
                    hole_score = hole_fill * 3.0
                    pillar_penalty = max_increase * 2.0
                    local_mean = float(np.mean(region)) if region.size else y
                    smooth_penalty = abs(new_local - local_mean) * 0.2
                    score = height_score + hole_score - pillar_penalty - smooth_penalty
                    scored.append((score, x, int(y)))

                scored.sort(key=lambda t: t[0], reverse=True)
                for i in range(min(top_k, len(scored))):
                    s, x, y = scored[i]
                    new_st = State(self.canvas_width)
                    new_st.height_map = st.height_map.copy()
                    new_st.placed = st.placed.copy()
                    new_st.score = st.score + s
                    new_st.max_height = st.max_height
                    new_local = y + h
                    new_st.height_map[x : x + w] = new_local
                    new_st.placed.append((x, y, w, h))
                    if new_local > new_st.max_height:
                        new_st.max_height = new_local
                    candidates.append(new_st)

            candidates.sort(key=lambda st: st.score, reverse=True)
            beam = candidates[:beam_size] if candidates else beam


        best = None
        best_eff = -1.0
        total_area = sum(w * h for w, h in rects)
        ideal_height = total_area / self.canvas_width
        for st in beam:
            max_h = int(st.max_height)
            eff = (ideal_height / max_h) if max_h > 0 else 0.0
            if eff > best_eff:
                best_eff = eff
                best = st

        if best is None:
            return self.solve(rects)

        bounding_area = self.canvas_width * int(best.max_height)
        area_utilization = (total_area / bounding_area) if bounding_area > 0 else 0.0
        return {
            "placed_rects": best.placed,
            "height_map": best.height_map,
            "max_height": int(best.max_height),
            "ideal_height": ideal_height,
            "efficiency": min(1.0, best_eff),
            "area_utilization": area_utilization,
            "total_area": total_area,
            "bounding_area": bounding_area,
        }


class ShelfHeuristic:

    def __init__(self, canvas_width: int = 100):
        self.canvas_width = canvas_width

    def solve(self, rects: List[Tuple[int, int]]) -> Dict:
        rects = list(rects)
  
        rects.sort(key=lambda r: (r[1], r[0]), reverse=True)

        placed: List[Tuple[int, int, int, int]] = []
        shelves: List[Dict] = []  

        current_y = 0
        for w, h in rects:
            placed_in_shelf = False
            for shelf in shelves:
                if shelf["remaining_width"] >= w and h <= shelf["height"]:
                    x = self.canvas_width - shelf["remaining_width"]
                    y = shelf["y"]
                    placed.append((x, y, w, h))
                    shelf["remaining_width"] -= w
                    placed_in_shelf = True
                    break

            if not placed_in_shelf:
                shelf_height = h
                shelves.append(
                    {
                        "y": current_y,
                        "height": shelf_height,
                        "remaining_width": self.canvas_width - w,
                    }
                )
                x = 0
                y = current_y
                placed.append((x, y, w, h))
                current_y += shelf_height

        max_height = current_y
        total_area = sum(w * h for w, h in rects)
        ideal_height = total_area / self.canvas_width
        efficiency = (ideal_height / max_height) if max_height > 0 else 0.0
        bounding_area = self.canvas_width * max_height
        area_utilization = (total_area / bounding_area) if bounding_area > 0 else 0.0

        return {
            "placed_rects": placed,
            "height_map": None,
            "max_height": max_height,
            "ideal_height": ideal_height,
            "efficiency": min(1.0, efficiency),
            "area_utilization": area_utilization,
            "total_area": total_area,
            "bounding_area": bounding_area,
        }
