#!/usr/bin/env python3
import re

with open('check_setup.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode symbols with text equivalents
replacements = {
    '🚀': '[ROCKET]',
    '🐍': '[PYTHON]',
    '✅': '[SUCCESS]',
    '❌': '[ERROR]',
    '📦': '[PACKAGES]',
    '🤖': '[AI]',
    '⚠️': '[WARNING]',
    '🎨': '[ART]',
    '📊': '[CHART]',
    '🎉': '[CELEBRATION]',
    '💡': '[TIP]'
}

for unicode_char, replacement in replacements.items():
    content = content.replace(unicode_char, replacement)

with open('check_setup.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Unicode symbols replaced in check_setup.py')
