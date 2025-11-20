import sys
import os
sys.path.append('reality_simulator')

from agency.network_decision_agent import ConsciousnessInterpreter, OllamaBridge

config = {'agency': {'vision_enabled': True, 'vision_model': 'gemma3:4b'}}
ollama_bridge = OllamaBridge()
interpreter = ConsciousnessInterpreter(ollama_bridge, config)

print(f'Language system exists: {hasattr(interpreter, "language_system")}')
if hasattr(interpreter, 'language_system'):
    vocab_size = len(interpreter.language_system.word_history['taught']) if hasattr(interpreter.language_system, 'word_history') else 0
    print(f'Vocabulary size: {vocab_size}')
    if hasattr(interpreter.language_system, 'word_history'):
        print(f'Word history keys: {list(interpreter.language_system.word_history.keys())}')
        if 'taught' in interpreter.language_system.word_history:
            print(f'Taught words: {interpreter.language_system.word_history["taught"][:10]}')  # First 10
else:
    print('No language system found')
