"""Lecture d'une instance de Zitzler et Thiele (format des fichiers de data/instances)."""
from typing import List, Tuple


class MOKPInstance:
    """Represents a Multi-Objective Knapsack Problem instance."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.n_objectives = 0
        self.n_items = 0
        self.capacities: List[float] = []
        self.weights: List[List[int]] = []
        self.profits: List[List[int]] = []
        self._load()

    def _load(self):
        with open(self.filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        header = lines[0].split()
        self.n_objectives = int(header[0])
        self.n_items = int(header[1])
        self.weights = [[] for _ in range(self.n_objectives)]
        self.profits = [[] for _ in range(self.n_objectives)]

        idx = 1
        for obj in range(self.n_objectives):
            self.capacities.append(float(lines[idx]))
            idx += 1
            for item in range(self.n_items):
                idx += 1  # skip item index
                self.weights[obj].append(int(lines[idx]))
                idx += 1
                self.profits[obj].append(int(lines[idx]))
                idx += 1

    def evaluate(self, solution: List[int]) -> Tuple[List[float], List[float]]:
        """Returns (objective_values, consumed_capacities) for a solution."""
        obj_values = [0.0] * self.n_objectives
        consumed = [0.0] * self.n_objectives
        for item in solution:
            for obj in range(self.n_objectives):
                obj_values[obj] += self.profits[obj][item]
                consumed[obj] += self.weights[obj][item]
        return obj_values, consumed

    def is_feasible(self, solution: List[int]) -> bool:
        """Check if solution respects all capacity constraints."""
        _, consumed = self.evaluate(solution)
        for obj in range(self.n_objectives):
            if consumed[obj] > self.capacities[obj]:
                return False
        return True

    def greedy_solution(self, weight_vector: List[float]) -> List[int]:
        """Build a feasible solution using weighted efficiency greedy heuristic."""
        ratios = []
        for i in range(self.n_items):
            weighted_profit = sum(weight_vector[obj] * self.profits[obj][i]
                                  for obj in range(self.n_objectives))
            total_weight = sum(self.weights[obj][i] for obj in range(self.n_objectives))
            if total_weight > 0:
                ratios.append((weighted_profit / total_weight, i))
            else:
                ratios.append((float('inf'), i))

        ratios.sort(reverse=True)
        solution = []
        consumed = [0.0] * self.n_objectives

        for _, item in ratios:
            can_add = True
            for obj in range(self.n_objectives):
                if consumed[obj] + self.weights[obj][item] > self.capacities[obj]:
                    can_add = False
                    break
            if can_add:
                solution.append(item)
                for obj in range(self.n_objectives):
                    consumed[obj] += self.weights[obj][item]
        return solution

    def get_top_items(self, n_top: int = 15, mode: str = "ratio") -> List[List[int]]:
        """Returns the top items for each objective based on ratio or raw profit."""
        top_items_per_obj = []
        for obj in range(self.n_objectives):
            if mode == "ratio":
                vals = [(self.profits[obj][i] / self.weights[obj][i]
                         if self.weights[obj][i] > 0 else 0, i)
                        for i in range(self.n_items)]
            else:
                vals = [(self.profits[obj][i], i) for i in range(self.n_items)]
            vals.sort(key=lambda x: x[0], reverse=True)
            top_items_per_obj.append([x[1] for x in vals[:n_top]])
        return top_items_per_obj
