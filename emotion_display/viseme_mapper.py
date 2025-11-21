"""
Viseme mapping for lip-sync animation.
Maps phonemes to mouth shapes (visemes) for realistic speech animation.

Based on the Preston Blair phoneme set commonly used in 2D animation.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re


class Viseme(Enum):
    """
    Standard viseme set for lip-sync animation.
    Based on Preston Blair's classic phoneme groups.
    """
    # Silence/Rest
    SILENCE = auto()       # Closed mouth, neutral

    # Vowels and consonants grouped by mouth shape
    AH = auto()           # A, I (as in "father", "hot")
    EE = auto()           # E, EE (as in "see", "eat")
    EH = auto()           # E (as in "bed", "pet")
    OH = auto()           # O (as in "go", "boat")
    OO = auto()           # U, OO (as in "too", "boot")

    # Consonants
    BMP = auto()          # B, M, P (lips together)
    FV = auto()           # F, V (teeth on lower lip)
    TH = auto()           # TH (tongue between teeth)
    LNT = auto()          # L, N, T, D (tongue behind teeth)
    KG = auto()           # K, G, NG (back of tongue)
    SZ = auto()           # S, Z (teeth together, slight opening)
    SH_CH = auto()        # SH, CH, J (lips pushed forward)
    R = auto()            # R (lips slightly rounded)
    W = auto()            # W (lips rounded, small opening)


@dataclass
class VisemeData:
    """Data defining a viseme's mouth shape."""
    open_amount: float     # 0-1, how open the mouth is
    width: float           # 0-1, mouth width (0.5 is neutral)
    pucker: float          # 0-1, lip rounding
    stretch: float         # 0-1, lip stretching
    upper_lip: float       # -1 to 1, upper lip position
    lower_lip: float       # -1 to 1, lower lip position

    def interpolate(self, other: 'VisemeData', t: float) -> 'VisemeData':
        """Interpolate between this viseme and another."""
        return VisemeData(
            open_amount=self.open_amount + (other.open_amount - self.open_amount) * t,
            width=self.width + (other.width - self.width) * t,
            pucker=self.pucker + (other.pucker - self.pucker) * t,
            stretch=self.stretch + (other.stretch - self.stretch) * t,
            upper_lip=self.upper_lip + (other.upper_lip - self.upper_lip) * t,
            lower_lip=self.lower_lip + (other.lower_lip - self.lower_lip) * t,
        )


# Viseme shape definitions
VISEME_SHAPES: Dict[Viseme, VisemeData] = {
    Viseme.SILENCE: VisemeData(
        open_amount=0.0, width=0.5, pucker=0.0, stretch=0.0,
        upper_lip=0.0, lower_lip=0.0
    ),
    Viseme.AH: VisemeData(
        open_amount=0.8, width=0.6, pucker=0.0, stretch=0.0,
        upper_lip=0.1, lower_lip=-0.2
    ),
    Viseme.EE: VisemeData(
        open_amount=0.3, width=0.75, pucker=0.0, stretch=0.7,
        upper_lip=0.0, lower_lip=0.0
    ),
    Viseme.EH: VisemeData(
        open_amount=0.4, width=0.6, pucker=0.0, stretch=0.3,
        upper_lip=0.0, lower_lip=-0.1
    ),
    Viseme.OH: VisemeData(
        open_amount=0.6, width=0.4, pucker=0.5, stretch=0.0,
        upper_lip=0.1, lower_lip=-0.1
    ),
    Viseme.OO: VisemeData(
        open_amount=0.4, width=0.3, pucker=0.8, stretch=0.0,
        upper_lip=0.15, lower_lip=0.0
    ),
    Viseme.BMP: VisemeData(
        open_amount=0.0, width=0.5, pucker=0.0, stretch=0.0,
        upper_lip=0.0, lower_lip=0.05
    ),
    Viseme.FV: VisemeData(
        open_amount=0.1, width=0.55, pucker=0.0, stretch=0.0,
        upper_lip=-0.1, lower_lip=0.1
    ),
    Viseme.TH: VisemeData(
        open_amount=0.15, width=0.55, pucker=0.0, stretch=0.1,
        upper_lip=0.0, lower_lip=-0.05
    ),
    Viseme.LNT: VisemeData(
        open_amount=0.2, width=0.55, pucker=0.0, stretch=0.2,
        upper_lip=0.0, lower_lip=-0.05
    ),
    Viseme.KG: VisemeData(
        open_amount=0.35, width=0.55, pucker=0.0, stretch=0.1,
        upper_lip=0.0, lower_lip=-0.1
    ),
    Viseme.SZ: VisemeData(
        open_amount=0.1, width=0.6, pucker=0.0, stretch=0.4,
        upper_lip=0.0, lower_lip=0.0
    ),
    Viseme.SH_CH: VisemeData(
        open_amount=0.2, width=0.4, pucker=0.4, stretch=0.0,
        upper_lip=0.1, lower_lip=0.0
    ),
    Viseme.R: VisemeData(
        open_amount=0.25, width=0.45, pucker=0.3, stretch=0.0,
        upper_lip=0.0, lower_lip=0.0
    ),
    Viseme.W: VisemeData(
        open_amount=0.2, width=0.35, pucker=0.6, stretch=0.0,
        upper_lip=0.1, lower_lip=0.0
    ),
}


# Phoneme to viseme mapping (ARPAbet phonemes)
PHONEME_TO_VISEME: Dict[str, Viseme] = {
    # Vowels
    'AA': Viseme.AH,    # father
    'AE': Viseme.EH,    # cat
    'AH': Viseme.AH,    # but
    'AO': Viseme.OH,    # dog
    'AW': Viseme.OH,    # cow
    'AY': Viseme.AH,    # bite
    'EH': Viseme.EH,    # bed
    'ER': Viseme.R,     # bird
    'EY': Viseme.EE,    # say
    'IH': Viseme.EH,    # bit
    'IY': Viseme.EE,    # beat
    'OW': Viseme.OH,    # boat
    'OY': Viseme.OH,    # boy
    'UH': Viseme.OO,    # book
    'UW': Viseme.OO,    # boot

    # Consonants
    'B': Viseme.BMP,
    'CH': Viseme.SH_CH,
    'D': Viseme.LNT,
    'DH': Viseme.TH,    # this
    'F': Viseme.FV,
    'G': Viseme.KG,
    'HH': Viseme.AH,    # hat (open mouth)
    'JH': Viseme.SH_CH, # judge
    'K': Viseme.KG,
    'L': Viseme.LNT,
    'M': Viseme.BMP,
    'N': Viseme.LNT,
    'NG': Viseme.KG,
    'P': Viseme.BMP,
    'R': Viseme.R,
    'S': Viseme.SZ,
    'SH': Viseme.SH_CH,
    'T': Viseme.LNT,
    'TH': Viseme.TH,    # think
    'V': Viseme.FV,
    'W': Viseme.W,
    'Y': Viseme.EE,
    'Z': Viseme.SZ,
    'ZH': Viseme.SH_CH, # measure

    # Silence
    'SIL': Viseme.SILENCE,
    '': Viseme.SILENCE,
}


# Simple character-based mapping for when phonemes aren't available
CHAR_TO_VISEME: Dict[str, Viseme] = {
    'a': Viseme.AH,
    'b': Viseme.BMP,
    'c': Viseme.KG,
    'd': Viseme.LNT,
    'e': Viseme.EE,
    'f': Viseme.FV,
    'g': Viseme.KG,
    'h': Viseme.AH,
    'i': Viseme.EE,
    'j': Viseme.SH_CH,
    'k': Viseme.KG,
    'l': Viseme.LNT,
    'm': Viseme.BMP,
    'n': Viseme.LNT,
    'o': Viseme.OH,
    'p': Viseme.BMP,
    'q': Viseme.KG,
    'r': Viseme.R,
    's': Viseme.SZ,
    't': Viseme.LNT,
    'u': Viseme.OO,
    'v': Viseme.FV,
    'w': Viseme.W,
    'x': Viseme.KG,
    'y': Viseme.EE,
    'z': Viseme.SZ,
    ' ': Viseme.SILENCE,
    '.': Viseme.SILENCE,
    ',': Viseme.SILENCE,
    '!': Viseme.SILENCE,
    '?': Viseme.SILENCE,
}


class VisemeMapper:
    """
    Maps text/phonemes to viseme sequences for lip-sync animation.
    """

    def __init__(self, use_phonemes: bool = False):
        """
        Initialize the viseme mapper.

        Args:
            use_phonemes: If True, expects ARPAbet phoneme input.
                         If False, uses simple character mapping.
        """
        self.use_phonemes = use_phonemes
        self._current_viseme = Viseme.SILENCE
        self._target_viseme = Viseme.SILENCE
        self._transition_progress = 1.0

    def get_viseme_shape(self, viseme: Viseme) -> VisemeData:
        """Get the shape data for a viseme."""
        return VISEME_SHAPES.get(viseme, VISEME_SHAPES[Viseme.SILENCE])

    def text_to_visemes(self, text: str) -> List[Tuple[Viseme, float]]:
        """
        Convert text to a sequence of visemes with durations.

        Args:
            text: Text to convert

        Returns:
            List of (viseme, duration) tuples
        """
        visemes = []
        text = text.lower()

        # Simple character-based mapping
        i = 0
        while i < len(text):
            char = text[i]

            # Check for digraphs
            if i + 1 < len(text):
                digraph = text[i:i+2]
                if digraph in ['th', 'sh', 'ch']:
                    if digraph == 'th':
                        visemes.append((Viseme.TH, 0.08))
                    elif digraph == 'sh':
                        visemes.append((Viseme.SH_CH, 0.08))
                    elif digraph == 'ch':
                        visemes.append((Viseme.SH_CH, 0.08))
                    i += 2
                    continue

            # Single character mapping
            if char in CHAR_TO_VISEME:
                viseme = CHAR_TO_VISEME[char]
                # Vowels get longer duration
                if viseme in [Viseme.AH, Viseme.EE, Viseme.EH, Viseme.OH, Viseme.OO]:
                    duration = 0.1
                elif viseme == Viseme.SILENCE:
                    duration = 0.05
                else:
                    duration = 0.06
                visemes.append((viseme, duration))

            i += 1

        # Add final silence
        if visemes and visemes[-1][0] != Viseme.SILENCE:
            visemes.append((Viseme.SILENCE, 0.1))

        return visemes

    def phonemes_to_visemes(self, phonemes: List[Tuple[str, float]]) -> List[Tuple[Viseme, float]]:
        """
        Convert phonemes with timing to visemes.

        Args:
            phonemes: List of (phoneme, duration) tuples

        Returns:
            List of (viseme, duration) tuples
        """
        visemes = []
        for phoneme, duration in phonemes:
            # Remove stress markers (numbers at end of ARPAbet phonemes)
            clean_phoneme = re.sub(r'\d+$', '', phoneme.upper())
            viseme = PHONEME_TO_VISEME.get(clean_phoneme, Viseme.SILENCE)
            visemes.append((viseme, duration))

        return visemes

    def get_speaking_visemes(self, intensity: float = 1.0) -> Viseme:
        """
        Get a viseme for generic speaking animation when no text is available.
        Cycles through common speaking shapes.

        Args:
            intensity: Speaking intensity (0-1)

        Returns:
            A viseme appropriate for speaking
        """
        import random
        speaking_visemes = [
            Viseme.AH, Viseme.EE, Viseme.OH, Viseme.EH,
            Viseme.LNT, Viseme.BMP, Viseme.SILENCE
        ]
        weights = [0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.15]

        return random.choices(speaking_visemes, weights=weights)[0]

    def interpolate_visemes(self, from_viseme: Viseme, to_viseme: Viseme,
                           progress: float) -> VisemeData:
        """
        Interpolate between two visemes.

        Args:
            from_viseme: Starting viseme
            to_viseme: Target viseme
            progress: Interpolation progress (0-1)

        Returns:
            Interpolated viseme data
        """
        from_data = self.get_viseme_shape(from_viseme)
        to_data = self.get_viseme_shape(to_viseme)

        # Use smooth interpolation (ease in/out)
        t = self._smooth_step(progress)

        return from_data.interpolate(to_data, t)

    @staticmethod
    def _smooth_step(t: float) -> float:
        """Smooth step function for natural transitions."""
        # Clamp to 0-1
        t = max(0.0, min(1.0, t))
        # Smooth step: 3t^2 - 2t^3
        return t * t * (3.0 - 2.0 * t)
