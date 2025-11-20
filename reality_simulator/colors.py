"""
🎨 COLOR SCHEME FOR REALITY SIMULATOR

ANSI color codes for visually organizing Reality Simulator output.
"""

class ColorScheme:
    """ANSI color codes for Reality Simulator output"""

    # Check if colors should be used
    _use_colors = True

    @classmethod
    def disable_colors(cls):
        """Disable color output (for terminals that don't support ANSI codes)"""
        cls._use_colors = False

    @classmethod
    def enable_colors(cls):
        """Enable color output"""
        cls._use_colors = True

    # Reset
    RESET = '\033[0m'

    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

    # Custom colors
    ORANGE = '\033[38;5;208m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

    # System component colors
    QUANTUM = BRIGHT_CYAN      # Quantum substrate
    EVOLUTION = BRIGHT_GREEN   # Genetic evolution
    NETWORK = BRIGHT_YELLOW    # Symbiotic network
    CONSCIOUSNESS = BRIGHT_MAGENTA  # Consciousness detector
    AGENCY = BRIGHT_BLUE       # AI Agency
    RENDERER = CYAN            # Reality renderer
    FEEDBACK = BRIGHT_RED      # Feedback controller

    # Special highlights
    SHARED_STATE = BRIGHT_YELLOW  # Shared state operations

    # Status colors
    SUCCESS = BRIGHT_GREEN
    WARNING = BRIGHT_YELLOW
    ERROR = BRIGHT_RED
    INFO = BRIGHT_WHITE

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text"""
        if not cls._use_colors:
            return text
        return f"{color}{text}{cls.RESET}"

    @classmethod
    def highlight_ollama(cls, text: str) -> str:
        """Legacy function - Ollama removed, returns text unchanged"""
        return text

    @classmethod
    def log_component(cls, component: str, message: str) -> str:
        """Colorize log messages by component"""
        if not cls._use_colors:
            return f"[{component.upper()}] {message}"

        component_colors = {
            'quantum': cls.QUANTUM,
            'evolution': cls.EVOLUTION,
            'network': cls.NETWORK,
            'consciousness': cls.CONSCIOUSNESS,
            'agency': cls.AGENCY,
            'renderer': cls.RENDERER,
            'feedback': cls.FEEDBACK,
            'shared': cls.SHARED_STATE,
        }

        color = component_colors.get(component.lower(), cls.INFO)
        return cls.colorize(f"[{component.upper()}] {message}", color)
