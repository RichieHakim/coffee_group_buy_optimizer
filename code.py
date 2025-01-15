coffee_packages = {
  'coco bongo' : [(0.25, 19.9), (1, 77)],
  'pink balloon' : [(0.25, 17.95), (1, 69.2)],
  'kumquat squat' : [(0.25, 16.95), (1, 65.2)],
  'the alchemist' : [(0.25, 14.95), (1, 57.2)],
  'banana split' : [(0.25, 21.95), (1, 85.2)],
  'big apple' : [(0.2, 29.95), (1, 117.2)],
  'tropic electric' : [(0.2, 22.95), (1, 112.15)],
  'rumba' : [(0.2, 25.6), (1, 125.4)],
  'darling peach' : [(0.2, 19), (1, 92.4)],
  'milky cake' : [(0.25, 19.25), (1, 74.4)],
  'currant bluff' : [(0.2, 18), (1, 87.4)],
  'magnolia' : [(0.2, 24.95), (1, 122.15)],
  'juniper' : [(0.25, 14.95), (1, 57.2)],
  'pineberry' : [(0.25, 17.95), (1, 69.2)],
  'lemon pearls' : [(0.25, 17), (1, 65.4)],
}

demands = {
  ('josh','coco bongo') : (0.125, 0.25, 0.125),
  ('josh','pink balloon') : (0.125, 0.25, 0.125),
  ('josh','kumquat squat') : (0, 0.25, 0.125),
  ('josh','the alchemist') : (0, 0.25, 0.125),
  ('josh','banana split') : (0, 0.125, 0),
  ('josh','big apple') : (0, 0.125, 0),
  ('josh','tropic electric') : (0, 0.1, 0),
  ('josh','rumba') : (0, 0.1, 0),
  ('josh','darling peach') : (0, 0.1, 0),
  ('josh', 'magnolia'): (0.075, 0.125, 0.1),

  ('rich', 'coco bongo'): (0.05, 0.125, 0.08),
  ('rich', 'banana split'): (0.05, 0.125, 0.08),
  ('rich', 'magnolia'): (0.05, 0.125, 0.08),

  ('cheushii', 'coco bongo'): (0.25, 0.25, 0.25),
  ('cheushii', 'magnolia'): (0.2, 0.2, 0.2),

  ('cessna', 'coco bongo'): (0.125, 0.25, 0.25),

  ('grim', 'milky cake'): (0.25, 0.25, 0.25),

  ('reuben', 'banana split'): (0.075, 0.25, 0.125),
  ('reuben', 'magnolia'): (0.075, 0.125, 0.125),
}

total_demands = {
  'josh': (0.25, 2.5, 0.5),
  'rich': (0.2, 0.25, 0.25),
  'reuben': (0.15, 0.4, 0.25),
}

alpha = 1 - 1e-2

units_size = 'kg'
units_money = 'EUR'



sol = optimize_coffee_purchase_with_totals(
    coffee_packages=coffee_packages,
    demands=demands,
    total_demands=total_demands,
    alpha=alpha
)
print(f"\n=== alpha={alpha} ===")
print("Status:", sol["status"])
print(f"Objective Value: {sol['objective_value']:.4f}")
print(f"Sum of normalized deviations: {sol['sum_devs_normalized']:.4f}")
print(f"Total Cost: {sol['total_cost']:.2f} {units_money}")

print("Packages Purchased:")
for (c, k), qty in sorted(sol["x_c_k"].items()):
    if qty > 0:
        size, cost = coffee_packages[c][k]
        print(f"  {str(c)+':':<15} {qty} × {size:.3f}kg @ {cost} {units_money}")

print("Allocations:")
for (n, c), grams in sorted(sol["a_n_c"].items()):
    if grams > 0:
        dev = sol["dev_n_c"][(n, c)]
        lb, ub, tgt = demands[(n, c)]
        print(f"  {str(n)+',':<10} {str(c)+',':<15} Alloc: {grams:.3f} kg "
              f"Constraints: ({lb}, {ub}, {tgt}), dev: {dev}")

print("Total Deviations:")
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
    print(f"  {str(n)+',':<10} TotalAlloc={sum_alloc:.3f} kg, DevTotal={devT:.3f} "
          f"Constraints: ({lbT}, {ubT}, {tgtT})")
