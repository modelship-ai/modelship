"""When the client is playing a session's spoken answer, as far as the session can tell. The audio goes out faster
than real time and the client plays it, so the session counts: each piece of audio plays for its length, from when it
is sent or when the piece before it ends, whichever is later."""

import math
import time
from collections.abc import Callable

# how long the room echoes an answer after its last audio plays
TAIL_SECONDS = 0.3


class Playback:
    """The client's playback of the session's audio, on *clock*."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        # when the audio sent so far ends
        self._until = -math.inf

    def sent(self, seconds: float) -> None:
        """Counts *seconds* of audio just sent."""
        self._until = max(self._until, self._clock()) + seconds

    def playing(self) -> bool:
        """Whether the client is playing audio sent, or the room still echoes it."""
        return self._clock() < self._until + TAIL_SECONDS

    def stop(self) -> None:
        """The client stopped playing."""
        self._until = -math.inf
