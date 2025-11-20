
import os

file_path = r"c:/Users/Shadow/Documents/Reality_Sim-main/reality_simulator/agency/network_decision_agent.py"

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines are 0-indexed in Python list, but 1-indexed in our analysis
# We want to keep lines 1 to 2254 (indices 0 to 2253)
# We want to skip lines 2255 to 4113 (indices 2254 to 4112)
# We want to keep lines 4114 to end (indices 4113 to end)

# Verify the split points
print(f"Line 2255 (index 2254): {lines[2254]}")
print(f"Line 4114 (index 4113): {lines[4113]}")

# Construct new content
new_lines = lines[:2254] + lines[4113:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("File updated successfully.")
