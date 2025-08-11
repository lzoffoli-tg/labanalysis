"""singlejumps test module"""

#! IMPORTS


__all__ = ["JumpTest"]


#! CLASSES


import pandas as pd

from ...frames.records.jumping import SingleJump
from ..protocols import Participant, TestProtocol


class JumpTest(TestProtocol):

    _jumps: list[SingleJump]

    @property
    def jumps(self):
        return self._jumps

    def add_jump(self, jump: SingleJump):
        if not isinstance(jump, SingleJump):
            raise ValueError("jump must be a  SingleJump instance.")
        self._jumps += [jump]

    def __init__(
        self,
        participant: Participant,
        normative_data_path: str = "",
        jumps: list[SingleJump] = [],
    ):
        if not isinstance(participant, Participant):
            raise ValueError("participant must be a Participant class instance.")
        if participant.weight is None:
            raise ValueError("participant's weight must be assigned.")
        self.set_participant(participant)
        self.set_normative_data_path(normative_data_path)
        self._jumps = []
        for jump in jumps:
            self.add_jump(jump)

    def copy(self):
        return JumpTest(
            participant=self.participant.copy(),
            normative_data_path=self.normative_data_path,
            jumps=self.jumps,
        )

    @property
    def results(self):
        analytics = []
        summary = []
        for i, jump in enumerate(self.jumps):

            # generate the analytics for the jumps
            df = jump.to_dataframe()
            df.insert(0, "Time", df.index)
            df.insert(0, "Jump", i + 1)
            analytics += [df]

            # add summary metrics
            metrics = jump.output_metrics
            metrics.insert(0, "jump", i + 1)
            summary += [metrics]

        # outcomes
        return {
            "summary": pd.concat(summary, ignore_index=True),
            "analytics": pd.concat(analytics, ignore_index=True),
        }
