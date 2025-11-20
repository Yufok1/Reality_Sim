#!/usr/bin/env python3
import re

with open('run_reality_simulator.bat', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode symbols with text equivalents for batch file
replacements = {
    '❌': '[ERROR]',
    '🔍': '[SEARCH]',
    '🎯': '[TARGET]',
    '👑': '[GOD]',
    '🔬': '[SCIENCE]',
    '🌟': '[STAR]',
    '🗣️': '[CHAT]',
    '⚙️': '[SETTINGS]',
    '🚀': '[ROCKET]',
    '🧪': '[TEST]',
    '📊': '[CHART]',
    '⚠️': '[WARNING]',
    '🔧': '[TOOL]',
    '👋': '[WAVE]'
}

for unicode_char, replacement in replacements.items():
    content = content.replace(unicode_char, replacement)

with open('run_reality_simulator.bat', 'w', encoding='utf-8') as f:
    f.write(content)

print('Unicode symbols replaced in run_reality_simulator.bat')
