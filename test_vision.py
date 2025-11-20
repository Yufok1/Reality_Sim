import sys
import os
sys.path.append('reality_simulator')

# Test vision analysis
from agency.network_decision_agent import ConsciousnessInterpreter, OllamaBridge

config = {'agency': {'vision_enabled': True, 'vision_model': 'gemma3:4b'}}
ollama_bridge = OllamaBridge()
interpreter = ConsciousnessInterpreter(ollama_bridge, config)

print(f'Vision enabled: {interpreter.vision_enabled}')

# Try a simple practice
if hasattr(interpreter, 'practice_language_generation'):
    print('Testing practice_language_generation...')
    try:
        result = interpreter.practice_language_generation(10, None, {})
        print(f'Practice result keys: {list(result.keys()) if result else "None"}')
    except Exception as e:
        print(f'Practice failed: {e}')
else:
    print('practice_language_generation method not found')
