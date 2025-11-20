#!/usr/bin/env python3
import re

with open('reality_simulator/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace common Unicode symbols with text equivalents
replacements = {
    '🚀': '[ROCKET]',
    '🌀': '[SPIRAL]',
    '🧬': '[DNA]',
    '🌐': '[NETWORK]',
    '🧠': '[BRAIN]',
    '🤖': '[ROBOT]',
    '🎨': '[ART]',
    '📡': '[SATELLITE]',
    '🌀': '[LATTICE]',
    '📊': '[CHART]',
    '⚠️': '[WARNING]',
    '🏁': '[FINISH]',
    '💾': '[SAVE]',
    '📂': '[FOLDER]',
    '🗣️': '[SPEECH]',
    '👤': '[USER]',
    '🤖': '[AI]',
    '🔄': '[CYCLE]',
    '⏹️': '[STOP]',
    '⏳': '[HOURGLASS]',
    '👋': '[WAVE]',
    '🎯': '[TARGET]',
    '🧠': '[CONSCIOUSNESS]',
    '⏱️': '[TIMER]',
    '🔧': '[TOOL]',
    '📋': '[CLIPBOARD]',
    '📈': '[GRAPH]',
    '🎚️': '[SLIDER]'
}

for unicode_char, replacement in replacements.items():
    content = content.replace(unicode_char, replacement)

with open('reality_simulator/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Unicode symbols replaced in main.py')
