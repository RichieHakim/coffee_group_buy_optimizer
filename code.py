import pulp
from typing import Dict, Tuple, List


def optimize_coffee_purchase_with_targets(
    coffee_packages: Dict[int, List[Tuple[float, float]]],
    demands: Dict[Tuple[int, int], Tuple[float, float, float]],
    solver_name: str = "PULP_CBC_CMD"
) -> Dict[str, dict]:
    """
    Minimize the total absolute deviation from a target amount of coffee for each 
    person–coffee pair, subject to lower and upper demand bounds. We allow fractional 
    splits of packages.

    RH 2025

    Args:
        coffee_packages (Dict[int, List[Tuple[float, float]]]):
            A mapping from coffee index c to a list of (size, cost) tuples, e.g.:
                {
                  0: [(100, 2.0), (500, 8.0)],
                  1: [(250, 3.0), (750, 9.0)]
                }
            Each tuple indicates a possible package size (grams) and a cost (not 
            used in the objective here, but used to ensure packages are discrete 
            increments).

        demands (Dict[Tuple[int, int], Tuple[float, float, float]]):
            A mapping (person, coffee) -> (lower_bound, upper_bound, target),
            e.g. demands[(0, 1)] = (100, 200, 150) means:
                - Person 0, Coffee 1
                - Must have between 100g and 200g
                - Target is 150g

        solver_name (str, optional):
            Which solver to use (passed to pulp). Defaults to "PULP_CBC_CMD".

    Returns:
        Dict[str, dict]:
            A dictionary with keys:
            - "x_c_k":  dict {(c, k): int} => how many packages of type k are purchased for coffee c
            - "a_n_c":  dict {(n, c): float} => allocated grams of coffee c to person n
            - "diff":   dict {(n, c): float} => the solver's absolute deviation variable for (n,c)
            - "total_deviation": float => the minimized sum of absolute deviations
            - "status": str => solver status (e.g., "Optimal", "Infeasible", etc.)
    """

    # ---------------------------------------------------
    # 1. Identify sets of coffees and people from demands
    # ---------------------------------------------------
    coffees = sorted(coffee_packages.keys())
    people = sorted({pc[0] for pc in demands.keys()})

    # Create the problem
    problem = pulp.LpProblem("GroupCoffeeBuyWithTargets", pulp.LpMinimize)

    # ---------------------------------------------------
    # 2. Decision variables
    # ---------------------------------------------------
    # x_c_k: integer variable for how many packages of type k to buy for coffee c
    x_c_k = {}
    for c in coffees:
        for k, (pkg_size, pkg_cost) in enumerate(coffee_packages[c]):
            var_name = f"x_{c}_{k}"
            x_c_k[(c, k)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)

    # a_n_c: continuous variable for grams allocated to person n of coffee c
    a_n_c = {}
    for n in people:
        for c in coffees:
            var_name = f"a_{n}_{c}"
            a_n_c[(n, c)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpContinuous)

    # diff_n_c: continuous variable representing the absolute deviation from target
    # We will enforce diff_n_c >= |a_n_c - target_n_c|
    diff_n_c = {}
    for n in people:
        for c in coffees:
            var_name = f"diff_{n}_{c}"
            diff_n_c[(n, c)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpContinuous)

    # ---------------------------------------------------
    # 3. Objective: minimize total deviation from targets
    # ---------------------------------------------------
    problem += pulp.lpSum(diff_n_c[(n, c)] for n in people for c in coffees), "Total_Absolute_Deviation"

    # ---------------------------------------------------
    # 4. Constraints
    # ---------------------------------------------------
    # (A) Coffee supply vs. allocated
    for c in coffees:
        # sum of purchased grams >= sum of allocated grams across all people
        problem += (
            pulp.lpSum(coffee_packages[c][k][0] * x_c_k[(c, k)]
                       for k, (pkg_size, pkg_cost) in enumerate(coffee_packages[c]))
            >=
            pulp.lpSum(a_n_c[(n, c)] for n in people),
            f"SupplyVsAlloc_c{c}"
        )

    # (B) Demand constraints: L_{n,c} <= a_{n,c} <= U_{n,c}
    for (n, c), (lb, ub, tgt) in demands.items():
        problem += (a_n_c[(n, c)] >= lb, f"LB_{n}_{c}")
        problem += (a_n_c[(n, c)] <= ub, f"UB_{n}_{c}")

    # (C) Absolute deviation constraints:
    #     diff_n_c >= a_n_c - target
    #     diff_n_c >= target - a_n_c
    # This ensures diff_n_c >= |a_n_c - target|
    for (n, c), (lb, ub, tgt) in demands.items():
        problem += (a_n_c[(n, c)] - tgt <= diff_n_c[(n, c)], f"DiffPos_{n}_{c}")
        problem += (tgt - a_n_c[(n, c)] <= diff_n_c[(n, c)], f"DiffNeg_{n}_{c}")

    # ---------------------------------------------------
    # 5. Solve
    # ---------------------------------------------------
    if solver_name == "PULP_CBC_CMD":
        solver = pulp.PULP_CBC_CMD(msg=False)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False)  # customize other solvers if needed

    problem.solve(solver)

    # ---------------------------------------------------
    # 6. Return results
    # ---------------------------------------------------
    solution_x = {(c, k): int(pulp.value(x_c_k[(c, k)]) or 0)
                  for c in coffees
                  for k in range(len(coffee_packages[c]))}
    solution_a = {(n, c): float(pulp.value(a_n_c[(n, c)]) or 0.0)
                  for n in people
                  for c in coffees}
    solution_diff = {(n, c): float(pulp.value(diff_n_c[(n, c)]) or 0.0)
                     for n in people
                     for c in coffees}

    return {
        "x_c_k": solution_x,
        "a_n_c": solution_a,
        "diff":  solution_diff,
        "total_deviation": pulp.value(problem.objective),
        "status": pulp.LpStatus[problem.status]
    }


def test_target_scenario():
    """
    A small demonstration of the target-based approach.
    
    - We'll have 2 coffees: coffee 0 has 2 possible package sizes, coffee 1 has 1 possible size.
    - We'll have 2 people, each with (lower_bound, upper_bound, target) for each coffee.
    - We'll see how well the solver can match the targets without cost constraints.
    """
    coffee_packages = {
        0: [(100, 2.0), (500, 9.0)],  # coffee 0: (size=100g,cost=2.0), (size=500g,cost=9.0)
        1: [(250, 4.0)]              # coffee 1: (size=250g,cost=4.0)
    }
    
    # For demands: demands[(person, coffee)] = (lower, upper, target)
    demands = {
        # Person 0 wants between 50g and 600g of coffee 0, ideally 300g
        (0, 0): (50, 600, 300),
        # Person 0 wants between 0g and 250g of coffee 1, ideally 100g
        (0, 1): (0, 250, 100),
        # Person 1 wants between 300g and 500g of coffee 0, ideally 400g
        (1, 0): (300, 500, 400),
        # Person 1 wants between 100g and 250g of coffee 1, ideally 200g
        (1, 1): (100, 250, 200),
    }

    sol = optimize_coffee_purchase_with_targets(coffee_packages, demands)

    print("=== Target Scenario Test ===")
    print(f"Status: {sol['status']}")
    print(f"Total Deviation: {sol['total_deviation']:.2f}")
    print("Packages Purchased (x_c_k):")
    for (c, k) in sorted(sol["x_c_k"].keys()):
        qty = sol["x_c_k"][(c, k)]
        if qty > 0:
            size, cost = coffee_packages[c][k]
            print(f"  Coffee {c}, Package {k}: {qty} x {size}g (cost={cost})")

    print("\nAllocations and Deviation:")
    for (n, c) in sorted(sol["a_n_c"].keys()):
        (lb, ub, tgt) = demands[(n, c)]
        alloc = sol["a_n_c"][(n, c)]
        diff = sol["diff"][(n, c)]
        print(f"  Person {n}, Coffee {c}: allocated={alloc:.1f}g, target={tgt}, diff={diff:.1f} (range {lb}..{ub})")

    print("")


if __name__ == "__main__":  
  coffee_packages = {
      'putushio': [(1.0, 10.0),],   # Coffee 0: size=1.0, price=10.0
      'halo': [(1.0, 10.0),],    
      'kenyan': [(1.0, 10.0),],    
  }
  demands = {
      ('rich', 'putushio'): (0.25, 1.0, 0.75),  # Person: 'rich' wants 0.25 to 1.0, but ideally 0.75 units of 'putushio'
      ('rich', 'halo'): (0.25, 0.5, 0.5),  
      ('rich', 'kenyan'): (0.0, 0.0, 0.0),
      ('reuben', 'putushio'): (0.25, 1.0, 0.25),  
      ('reuben', 'halo'): (0.5, 1.0, 0.75),
      ('reuben', 'kenyan'): (0.0, 0.0, 0.0),
      ('kyle', 'putushio'): (0.25, 0.5, 0.5),
      ('kyle', 'halo'): (0.5, 1.0, 0.75),
      ('kyle', 'kenyan'): (1.0, 1.0, 1.0),
      ('luke', 'putushio'): (0.25, 0.5, 0.5),
      ('luke', 'halo'): (0.5, 1.0, 0.5),
      ('luke', 'kenyan'): (1.0, 1.0, 1.0),
  }

  ## convert str keys to unique integers
  idx_coffees = {coffee: i for i, coffee in enumerate(coffee_packages.keys())}
  unique_people = sorted(list({pc[0] for pc in demands.keys()}))
  idx_people = {person: i for i, person in enumerate(unique_people)}

  coffee_packages = {idx_coffees[coffee]: sizes for coffee, sizes in coffee_packages.items()}
  demands = {(idx_people[keys[0]], idx_coffees[keys[1]]): values for keys, values in demands.items()}
  # print(coffee_packages)
  # print(demands)

  sol = optimize_coffee_purchase_with_targets(coffee_packages, demands)
  print("=== Test #1: Small Scenario ===")
  print(f"Status: {sol['status']}")
  # print(f"Total Cost: {sol['total_cost']} money")
  print("Packages purchased (x_c_k):")
  for (c, k), qty in sorted(sol["x_c_k"].items()):
      if qty > 0:
          size_cost = coffee_packages[c][k]
          print(f"  Coffee {c}, Package {k} => {qty} units of {size_cost[0]} units @ {size_cost[1]} money")
  print("Allocations (a_n_c):")
  for (n, c), grams in sorted(sol["a_n_c"].items()):
      lower_bound, upper_bound, target = demands[(n, c)]
      print(f"  Person {n}, Coffee {c}: {grams:.2f} units (wanted {lower_bound} - {upper_bound}, target: {target})")
  print("")
