#!/usr/bin/env python3
"""
Reality Simulator Model Update Script
Updates config.json and Python defaults with selected Ollama models
"""

import json
import re
import sys
import os

def update_config_json(ai_model, vision_model):
    """Update the permanent config.json with selected models"""
    config_path = "config.json"

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Update agency models
        if 'agency' not in config:
            config['agency'] = {}

        config['agency']['ai_model'] = ai_model
        config['agency']['vision_model'] = vision_model

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[SUCCESS] Updated {config_path} with AI: {ai_model}, Vision: {vision_model}")
        return True

    except Exception as e:
        print(f"[ERROR] Error updating config.json: {e}")
        return False

def update_python_file(filepath, ai_model, vision_model):
    """Update hardcoded model references in a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        original_content = content

        # Update AI model references (single and double quotes)
        content = re.sub(r"'gemma3:4b'", f"'{ai_model}'", content)
        content = re.sub(r'"gemma3:4b"', f'"{ai_model}"', content)

        # Update vision model references
        content = re.sub(r"'gemma3:4b'", f"'{vision_model}'", content)
        content = re.sub(r'"gemma3:4b"', f'"{vision_model}"', content)

        # Special case for network_decision_agent.py comments
        content = re.sub(r'gemma3:4b model for binary', f'{ai_model} model for binary', content)
        content = re.sub(r'Reasonable timeout for gemma3:4b', f'Reasonable timeout for {ai_model}', content)
        content = re.sub(r'Default Ollama model', f'Default Ollama model ({ai_model})', content)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[SUCCESS] Updated {filepath}")
            return True
        else:
            print(f"[INFO] No changes needed in {filepath}")
            return False

    except Exception as e:
        print(f"[ERROR] Error updating {filepath}: {e}")
        return False

def update_check_setup(ai_model, vision_model):
    """Update check_setup.py with selected models"""
    try:
        with open("check_setup.py", 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Update model verification
        content = re.sub(r'gemma3:4b.*- OK', f'{ai_model} - OK', content)

        # Update pull instructions
        content = re.sub(r'ollama pull gemma3:4b', f'ollama pull {ai_model}', content)

        with open("check_setup.py", 'w', encoding='utf-8') as f:
            f.write(content)

        print("[SUCCESS] Updated check_setup.py")
        return True

    except Exception as e:
        print(f"[ERROR] Error updating check_setup.py: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python update_models.py <ai_model> <vision_model>")
        sys.exit(1)

    ai_model = sys.argv[1]
    vision_model = sys.argv[2]

    print(f"[UPDATE] Updating Reality Simulator models...")
    print(f"   AI Model: {ai_model}")
    print(f"   Vision Model: {vision_model}")
    print()

    # Update config.json
    config_updated = update_config_json(ai_model, vision_model)

    # Update Python files
    python_files = [
        "reality_simulator/main.py",
        "reality_simulator/agency/network_decision_agent.py",
        "reality_simulator/agency/agency_router.py"
    ]

    python_updated = False
    for filepath in python_files:
        if os.path.exists(filepath):
            if update_python_file(filepath, ai_model, vision_model):
                python_updated = True

    # Update check_setup.py
    setup_updated = update_check_setup(ai_model, vision_model)

    print()
    if config_updated:
        print("[SUCCESS] Model configuration updated successfully!")
        print("   - config.json: [SUCCESS] Updated")
        print("   - Python defaults: [SUCCESS] Updated" if python_updated else "   - Python defaults: [INFO] No changes needed")
        print("   - Setup verification: [SUCCESS] Updated" if setup_updated else "   - Setup verification: [ERROR] Failed")
    else:
        print("[ERROR] Failed to update configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
