import pulp
from typing import Dict, Tuple, List, Optional


def optimize_coffee_purchase_with_totals(
    coffee_packages: Dict[str, List[Tuple[float, float]]],
    demands: Dict[Tuple[str, str], Tuple[float, float, float]],
    total_demands: Optional[Dict[str, Tuple[float, float, float]]] = None,
    alpha: float = 0.5,
    solver_name: str = "PULP_CBC_CMD"
) -> Dict[str, dict]:
    """
    MILP that combines:
      (1) Per-coffee demands
      (2) Total coffee demand (summed over all coffees)
    in a single objective that balances:
        - cost
        - normalized deviations (per coffee and overall)

    Objective:|
        MINIMIZE alpha * SUM( dev_{n,c}/UB_{n,c} + dev_total_n / UB_{n}^{total} )
                + (1 - alpha) * SUM( cost_{c,k} * x_{c,k} )

    Constraints:
      - No leftovers: sum of purchased packages == sum of allocations (for each coffee).
      - Per-coffee demands: LB_{n,c} <= a_{n,c} <= UB_{n,c}, dev_{n,c} >= | a_{n,c} - T_{n,c} |
      - Total demands: LB_{n}^{total} <= sum_{c} a_{n,c} <= UB_{n}^{total},
                       dev_total_n >= | sum_{c} a_{n,c} - T^{total}_n |

    QoL Enhancements:
      - If total_demands is not provided or missing entries, it defaults to the sum of per-coffee demands.
      - Missing (person, coffee) entries in demands are filled with (0, 0, 0).

    Args:
        coffee_packages (Dict[str, List[Tuple[float, float]]]):
            For each coffee name, a list of (pkg_size, pkg_cost).
            e.g. {'coco bongo': [(0.25, 19.9), (1.0, 77.0)], ...}
        demands (Dict[Tuple[str, str], Tuple[float, float, float]]):
            (person, coffee) -> (lower_bound, upper_bound, target).
            e.g. {('josh','coco bongo'): (0.125, 0.25, 0.125), ...}
        total_demands (Optional[Dict[str, Tuple[float, float, float]]]):
            person -> (lower_bound, upper_bound, target) for the total coffee across all varieties.
            e.g. {'josh': (0.25, 0.5, 0.5), ...}
            If None or missing entries, defaults are computed as sums of per-coffee demands.
        alpha (float):
            Weight [0..1] on normalized deviation. (1 - alpha) is the weight on cost.
        solver_name (str):
            Name of the solver. Defaults to 'PULP_CBC_CMD'.

    Returns:
        A dict with:
            - "x_c_k":        how many packages of each type are purchased
            - "a_n_c":        actual allocated kg of coffee
            - "dev_n_c":      deviation per coffee-person
            - "dev_total_n":  total coffee deviation per person
            - "objective_value": final objective
            - "sum_devs_normalized": sum of (dev_{n,c}/UB_{n,c} + dev_total_n / UB_{n}^{total})
            - "total_cost": total cost of purchased packages
            - "status": solver status (Optimal, Infeasible, etc.)
    """

    # 1. Validate alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    # 2. Handle Missing Inputs and Defaults

    # Initialize total_demands if not provided
    if total_demands is None:
        total_demands = {}

    # Identify all coffees
    coffees = sorted(coffee_packages.keys())

    # Identify all people from demands and total_demands
    people_from_demands = {p for (p, c) in demands.keys()}
    people_from_totals = set(total_demands.keys())
    all_people = sorted(people_from_demands.union(people_from_totals))

    # Fill in missing (person, coffee) entries with (0, 0, 0)
    complete_demands = {}
    for person in all_people:
        for coffee in coffees:
            key = (person, coffee)
            if key in demands:
                complete_demands[key] = demands[key]
            else:
                complete_demands[key] = (0.0, 0.0, 0.0)  # Default to no demand

    # Update demands to the complete set
    demands = complete_demands

    # Compute total_demands for people not specified
    for person in all_people:
        if person not in total_demands:
            # Sum of lower bounds, upper bounds, and targets from per-coffee demands
            sum_lb = sum(
                demands[(person, coffee)][0] for coffee in coffees
            )
            sum_ub = sum(
                demands[(person, coffee)][1] for coffee in coffees
            )
            sum_tgt = sum(
                demands[(person, coffee)][2] for coffee in coffees
            )
            total_demands[person] = (sum_lb, sum_ub, sum_tgt)

    # 3. Create the optimization problem
    problem = pulp.LpProblem("CoffeeAllocationWithTotals", pulp.LpMinimize)

    # 4. Decision variables

    # x_c_k: integer number of packages purchased for coffee c, package type k
    x_c_k = {}
    for c in coffees:
        for k, (pkg_size, pkg_cost) in enumerate(coffee_packages[c]):
            var_name = f"x_{c}_{k}"
            x_c_k[(c, k)] = pulp.LpVariable(var_name, lowBound=0, cat=pulp.LpInteger)

    # a_n_c: allocated kg of coffee c to person n
    a_n_c = {}
    # dev_n_c: deviation from target for each coffee-person
    dev_n_c = {}

    # dev_total_n: overall coffee deviation for each person n
    dev_total_n = {}

    # Populate variables based on complete_demands
    for (n, c), (lb, ub, tgt) in demands.items():
        a_n_c[(n, c)] = pulp.LpVariable(f"a_{n}_{c}", lowBound=0, cat=pulp.LpContinuous)
        dev_n_c[(n, c)] = pulp.LpVariable(f"dev_{n}_{c}", lowBound=0, cat=pulp.LpContinuous)

    # Populate dev_total_n variables
    for n, (lbT, ubT, tgtT) in total_demands.items():
        dev_total_n[n] = pulp.LpVariable(f"dev_total_{n}", lowBound=0, cat=pulp.LpContinuous)

    # 5. Objective function

    # Cost
    total_cost = pulp.lpSum(
        coffee_packages[c][k][1] * x_c_k[(c, k)]
        for c in coffees
        for k in range(len(coffee_packages[c]))
    )

    # Sum of per-coffee deviations (normalized by each person's coffee UB_{n,c})
    sum_devs_per_coffee = []
    for (n, c), (lb, ub, tgt) in demands.items():
        # dev_n_c[(n,c)] * (1.0 / ub) if UB_{n,c} > 0
        if ub > 0:
            sum_devs_per_coffee.append(dev_n_c[(n, c)] * (1.0 / ub))
        else:
            # If UB=0, and target=0 (since a_n_c <= 0), ensure dev_n_c=0
            # Can add constraints if necessary, but here we skip
            pass

    # Sum of total-coffee deviations (normalized by UB^total_{n})
    sum_devs_total = []
    for n, (lbT, ubT, tgtT) in total_demands.items():
        if ubT > 0:
            sum_devs_total.append(dev_total_n[n] * (1.0 / ubT))
        else:
            # Similar to above, handle if UB^total_n = 0
            pass

    sum_devs_normalized = pulp.lpSum(sum_devs_per_coffee) + pulp.lpSum(sum_devs_total)

    # Define the objective
    problem += alpha * sum_devs_normalized + (1 - alpha) * total_cost, "Cost_And_Deviation"

    # 6. Constraints

    # (A) No leftovers: sum of purchased == sum of allocations, for each coffee
    for c in coffees:
        total_purchased = pulp.lpSum(
            coffee_packages[c][k][0] * x_c_k[(c, k)]
            for k in range(len(coffee_packages[c]))
        )
        total_allocated = pulp.lpSum(
            a_n_c.get((n, c), 0.0) for n in all_people
        )
        problem += (total_purchased == total_allocated, f"NoLeftover_{c}")

    # (B) Per-coffee demand constraints: LB_{n,c} <= a_{n,c} <= UB_{n,c}
    #     and dev_n_c >= | a_{n,c} - T_{n,c} |
    for (n, c), (lb, ub, tgt) in demands.items():
        # Allocation bounds
        problem += (a_n_c[(n, c)] >= lb, f"LB_{n}_{c}")
        problem += (a_n_c[(n, c)] <= ub, f"UB_{n}_{c}")

        # Deviation definitions
        problem += (dev_n_c[(n, c)] >= a_n_c[(n, c)] - tgt, f"DevPos_{n}_{c}")
        problem += (dev_n_c[(n, c)] >= tgt - a_n_c[(n, c)], f"DevNeg_{n}_{c}")

    # (C) Total coffee constraints for each person: LB^total_n <= sum_c a_{n,c} <= UB^total_n
    #     and dev_total_n >= | sum_c a_{n,c} - T^{total}_n |
    for n, (lbT, ubT, tgtT) in total_demands.items():
        sum_alloc_n = pulp.lpSum(
            a_n_c.get((n, c), 0.0) for c in coffees
        )

        # Total allocation bounds
        problem += (sum_alloc_n >= lbT, f"LB_total_{n}")
        problem += (sum_alloc_n <= ubT, f"UB_total_{n}")

        # Deviation definitions for total allocation
        problem += (dev_total_n[n] >= sum_alloc_n - tgtT, f"DevPos_total_{n}")
        problem += (dev_total_n[n] >= tgtT - sum_alloc_n, f"DevNeg_total_{n}")

    # 7. Solve the problem
    if solver_name == "PULP_CBC_CMD":
        solver = pulp.PULP_CBC_CMD(msg=False)
    else:
        # Default to CBC if unknown solver_name is provided
        solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]

    # 8. Gather results

    # How many packages purchased
    solution_x = {}
    for (c, k), var in x_c_k.items():
        val = pulp.value(var)
        solution_x[(c, k)] = int(round(val)) if val is not None else 0

    # Allocation per coffee-person
    solution_a = {}
    for (n, c), var in a_n_c.items():
        alloc = pulp.value(var)
        solution_a[(n, c)] = round(alloc, 4) if alloc is not None else 0.0

    # Deviation per coffee-person
    solution_dev = {}
    for (n, c), var in dev_n_c.items():
        dev = pulp.value(var)
        solution_dev[(n, c)] = round(dev, 4) if dev is not None else 0.0

    # Total deviation per person
    solution_dev_total = {}
    for n, var in dev_total_n.items():
        devT = pulp.value(var)
        solution_dev_total[n] = round(devT, 4) if devT is not None else 0.0

    # Objective value
    obj_val = pulp.value(problem.objective)

    # Re-compute sum_devs_normalized
    sum_devs_norm = 0.0
    for (n, c), (lb, ub, tgt) in demands.items():
        if ub > 0:
            sum_devs_norm += solution_dev[(n, c)] / ub
    for n, (lbT, ubT, tgtT) in total_demands.items():
        if ubT > 0:
            sum_devs_norm += solution_dev_total[n] / ubT

    # Total cost
    total_cost_val = 0.0
    for (c, k), qty in solution_x.items():
        pkg_cost = coffee_packages[c][k][1]
        total_cost_val += pkg_cost * qty

    return {
        "x_c_k": solution_x,
        "a_n_c": solution_a,
        "dev_n_c": solution_dev,
        "dev_total_n": solution_dev_total,
        "objective_value": obj_val,
        "sum_devs_normalized": sum_devs_norm,
        "total_cost": total_cost_val,
        "status": status
    }


def demo_total_demands_qol():
    """
    Demonstration of the coffee allocation with QoL enhancements:
      - total_demands defaults to sum of per-coffee demands if not provided.
      - missing (person, coffee) entries are filled with (0, 0, 0).
    Runs the optimizer with different alpha values to show the trade-off between
    cost minimization and deviation minimization.
    """
    # Define coffee names and packages
    coffee_names = [
        "coco bongo", "pink balloon", "kumquat squat",
        "the alchemist", "banana split", "big apple",
        "tropic electric", "rumba", "darling peach",
        "milky cake", "currant bluff", "magnolia",
        "juniper", "pineberry", "lemon pearls"
    ]

    coffee_packages = {
        'coco bongo': [(0.25, 19.9), (1.0, 77.0)],
        'pink balloon': [(0.25, 17.95), (1.0, 69.2)],
        'kumquat squat': [(0.25, 16.95), (1.0, 65.2)],
        'the alchemist': [(0.25, 14.95), (1.0, 57.2)],
        'banana split': [(0.25, 21.95), (1.0, 85.2)],
        'big apple': [(0.2, 29.95), (1.0, 117.2)],
        'tropic electric': [(0.2, 22.95), (1.0, 112.15)],
        'rumba': [(0.2, 25.6), (1.0, 125.4)],
        'darling peach': [(0.2, 19), (1.0, 92.4)],
        'milky cake': [(0.25, 19.25), (1.0, 74.4)],
        'currant bluff': [(0.2, 18), (1.0, 87.4)],
        'magnolia': [(0.2, 24.95), (1.0, 122.15)],
        'juniper': [(0.25, 14.95), (1.0, 57.2)],
        'pineberry': [(0.25, 17.95), (1.0, 69.2)],
        'lemon pearls': [(0.25, 17), (1.0, 65.4)],
    }

    # Define per-coffee demands: (person, coffee) -> (LB, UB, Target)
    demands = {
        ('josh', 'coco bongo'): (0.125, 0.25, 0.125),
        ('josh', 'pink balloon'): (0.125, 0.25, 0.125),
        ('josh', 'kumquat squat'): (0.0, 0.25, 0.125),
        ('josh', 'the alchemist'): (0.0, 0.25, 0.125),
        ('josh', 'banana split'): (0.0, 0.125, 0.0),
        ('josh', 'big apple'): (0.0, 0.125, 0.0),
        ('josh', 'tropic electric'): (0.0, 0.1, 0.0),
        ('josh', 'rumba'): (0.0, 0.1, 0.0),
        ('josh', 'darling peach'): (0.0, 0.1, 0.0),

        ('max', 'coco bongo'): (2.0, 10.0, 4.0),

        ('rich', 'coco bongo'): (0.08, 0.125, 0.125),
        ('rich', 'banana split'): (0.08, 0.125, 0.125),

        ('cheushii', 'coco bongo'): (0.25, 0.25, 0.25),
        ('cheushii', 'magnolia'): (0.2, 0.2, 0.2),

        ('cessna', 'coco bongo'): (0.125, 0.25, 0.25),

        ('grim', 'milky cake'): (0.25, 0.25, 0.25),

        ('reuben', 'banana split'): (0.0, 0.25, 0.125),
        ('reuben', 'magnolia'): (0.0, 0.125, 0.125),
    }

    # Define total demands: person -> (LB_total, UB_total, T_total)
    # Note: 'total_demands' is optional. Missing entries will be computed.
    # For demonstration, we'll omit some entries to test QoL features.
    total_demands = {
        'josh': (0.25, 0.75, 0.5),
        'max': (2.0, 10.0, 5.0),
        # 'rich' is intentionally omitted to test automatic computation
        # 'cheushii' is omitted
        # 'cessna' is omitted
        # 'grim' is omitted
        # 'reuben' is omitted
    }

    units_size = 'kg'
    units_money = 'EUR'

    # Run the optimizer with different alpha values
    for alpha in [0.0, 0.5, 0.9]:
        sol = optimize_coffee_purchase_with_totals(
            coffee_packages=coffee_packages,
            demands=demands,
            total_demands=total_demands,
            alpha=alpha
        )
        print("Status:", sol["status"])
        print(f"Objective Value: {sol['objective_value']:.4f}")
        print(f"Sum of normalized deviations: {sol['sum_devs_normalized']:.4f}")
        print(f"Total Cost: {sol['total_cost']:.2f} {units_money}")

        print("\n=== Inputs ===")
        print("Coffee Packages:")
        for name, options in coffee_packages.items():
            print(f"{name+':':<17} {''.join([f'{option[0]:.2f} {units_size} for {option[1]:<6.2f} {units_money} @ {option[1] / option[0]:<6.2f} {units_money}/{units_size},    ' for option in options])}")

        print("\nDemands:")
        print(f"__Name__   __Coffee__           __(low, high, target) {units_size}__")
        for (n, c), (lb, ub, tgt) in sorted(demands.items()):
            print(f"{str(n)+',':<10} {str(c)+',':<20} ({lb}, {ub}, {tgt})")

        print(f"\nTotal demands:")
        print(f"__Name__   __(low, high, target) {units_size}__")
        for n, (lbT, ubT, tgtT) in sorted(total_demands.items()):
            print(f"{str(n)+',':<10} ({lbT}, {ubT}, {tgtT})")

        print("\n=== Results ===")

        print("Packages Purchased:")
        print(f"  __Coffee__      __Amount__")
        for (c, k), qty in sorted(sol["x_c_k"].items()):
            if qty > 0:
                size, cost = coffee_packages[c][k]
                print(f"  {str(c)+':':<15} {qty} × {size:.3f} {units_size} @ {cost} {units_money}")

        print("\nAllocations:")
        print(f"  __Name__   __Coffee__        __Alloc__    __Dev__")
        for (n, c), grams in sorted(sol["a_n_c"].items()):
            if (n, c) in demands:
                if demands[(n, c)][2] > 0:
                    dev = sol["dev_n_c"][(n, c)]
                    lb, ub, tgt = demands[(n, c)]
                    print(f"  {str(n)+',':<10} {str(c)+',':<17} {grams:.3f} {units_size},    {dev} {units_size}")

        print("\nTotal Deviations:")
        print(f"  __Name__   __Total__   __Deviance__")
        for n in sorted(sol["dev_total_n"].keys()):
            devT = sol["dev_total_n"][n]
            lbT, ubT, tgtT = sol["dev_total_n"][n], 0, 0  # Initialize
            # Retrieve from total_demands or computed sum
            # Since total_demands includes all people after filling in
            # we need to get the actual total_demands used
            # However, in our implementation, all people have their total_demands filled in
            # So we can iterate through total_demands again
            lbT, ubT, tgtT = total_demands[n]
            sum_alloc = sum(sol["a_n_c"].get((n, c), 0.0) for c in coffee_packages.keys())
            print(f"  {str(n)+',':<10} {sum_alloc:.3f} {units_size},   {devT:.3f} {units_size} ")


if __name__ == "__main__":
    demo_total_demands_qol()
