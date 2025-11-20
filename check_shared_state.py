#!/usr/bin/env python3
import json, os

shared_file = os.path.join('data', '.shared_simulation_state.json')
if os.path.exists(shared_file):
    with open(shared_file, 'r') as f:
        data = json.load(f)
    print('Shared state file exists!')
    print(f'Keys: {list(data.keys())}')
    if 'data' in data and 'lattice' in data['data']:
        lattice = data['data']['lattice']
        print(f'Lattice data keys: {list(lattice.keys())}')
        print(f'Particles: {lattice.get("particles", "NOT FOUND")}')
    else:
        print('No lattice data in shared state')
        if 'data' in data:
            print(f'Data keys: {list(data["data"].keys())}')
else:
    print('Shared state file does NOT exist')
