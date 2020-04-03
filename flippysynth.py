#!/usr/bin/python
import time
import math
import random
import pyaudio
import numpy as np
from typing import Callable, Optional, List
from enum import Enum
import cython

STUFF = "HI"


class Device:
    pass


class Wave:
    # Function = Callable[[float], float]

    @staticmethod
    def SINE(f: float) -> float:
        return math.sin(math.pi * 2.0 * f)

    @staticmethod
    def TRIANGLE(f: float) -> float:
        return (
            -math.fmod(f, 1.0)
            if (math.fmod(f * 2.0, 1.0) > 0.5)
            else math.fmod(f + 0.25, 1.0)
        )

    @staticmethod
    def SQUARE(f: float) -> float:
        return 1.0 if math.fmod(f, 1.0) < 0.5 else -1.0

    @staticmethod
    def SAW(f: float) -> float:
        return math.fmod(f, 1.0) - 0.5

    @staticmethod
    def NOISE(f: float) -> float:
        return math.fmod(random.random(), 1.0)


FilterType = Enum("FilterType", "None LP HP BP")


# AudioCallback = Callable[[Optional[int], int, dict, int], [List[int], int]]


class Synth(Device):

    FRAMES: int = 512
    RATE: int = 44100
    CHANNELS: int = 1
    WIDTH: int = 2

    class FX:
        pass

    class Filter(FX):
        def __init__(self, t: FilterType = FilterType.LP):
            self.val: List[int] = [0.5, 0.01]
            self.names: List[str] = ["cutoff", "resonance"]
            self.type: int = t
            self.buf = np.zeros([4], dtype=np.int)
            for i in len(self.buf):
                self.buf[i] = 0

        def __call__(self, v):
            if self.type == FilterType.NONE:
                return v

            cutoff: float = self.val[0]
            res: float = self.val[1]

            cutoff: float = max(0.01, min(0.99, cutoff))
            feedback: float = res + res / (1.0 - cutoff)

            self.buf[0] += cutoff * (
                v - self.buf[0] + feedback * (self.buf[0] - self.buf[1])
            )
            self.buf[1] += cutoff * (self.buf[0] - self.buf[1])
            self.buf[2] += cutoff * (self.buf[1] - self.buf[2])
            self.buf[3] += cutoff * (self.buf[2] - self.buf[3])
            if self.type == FilterType.LP:
                return self.buf[1]
            elif self.type == FilterType.HP:
                return v - self.buf[-1]
            elif self.type == FilterType.BP:
                return self.buf[0] - self.buf[-1]

    class Oscillator:
        def __init__(
            self,
            synth,
            # func: Wave.Function = Wave.SQUARE,
            func=Wave.SQUARE,
            crush: int = 1,
            dest: bool = True,
        ):
            # self.default_func: Wave.Function = func
            self.default_func = func
            self.reset()
            self.rate_crush: int = crush
            self.synth = synth
            if dest:
                self.dest = Synth.Oscillator(synth, func, crush, False)

        def reset(self):
            self.rate_crush: int = 1
            self.amp: float = 0.0
            self.midinote: int = -1
            self.pitch: float = 0.0
            self.phase: float = 0.0
            self.vib_phase: float = 0.0
            self.mix: float = 1.0
            self.vib: float = 0.0
            # self.func: Wave.Function = self.default_func
            self.func = self.default_func
            self.t: float = 0
            self.next = False
            self.dest: Optional[Synth.Oscillator] = None
            self.sign = 0
            self.ch = 0
            self.end = False
            # self.bufs = None
            # self.fx = None
            # self.enabled = False

        def note(self, n=0, v=1.0, **kwargs):
            self.dest.func = kwargs.get("func", self.default_func)
            self.dest.midinote = n
            self.dest.pitch = Synth.midi_to_pitch(n)
            self.dest.amp = v
            self.next = True
            # print('note', n, v, self.dest.pitch)

        # def on(self):
        #     self.note(0, 1.0)
        def generate(self) -> None:
            pass

        def off(self) -> None:
            self.dest.amp: float = 0.0
            self.dest.midinote: int = -1
            self.next: bool = True

        def swap(self):
            nx: Optional[Synth.Oscillator] = self.dest
            self.dest = None
            self.reset()
            nx.dest = self
            # nx = self.dest
            # self.dest = None
            # self.reset()
            # nx.dest = self
            return nx

        def sample(self, n: int, block_recur: bool = False):
            osc = self
            v = osc.amp * osc.func(osc.phase) * osc.mix
            sgn = np.sign(v)
            if self.next and (sgn == 0 or (self.sign != 0 and sgn != self.sign)):
                if block_recur:
                    assert False
                    return osc, 0.0
                osc = osc.swap()
                osc.sign = sgn
                vib = math.sin(self.vib_phase)
                self.vib_phase += self.vib / Synth.RATE
                osc.phase += osc.pitch / Synth.RATE
                # osc.phase += osc.rate_crush * osc.pitch / Synth.RATE
                return osc.sample(True)
            osc.sign = sgn
            if not block_recur:
                vib = math.sin(self.vib_phase)
                self.vib_phase += self.vib / Synth.RATE
                osc.phase += (osc.pitch + vib) / Synth.RATE
                # osc.phase += osc.rate_crush * (osc.pitch+vib) / Synth.RATE
            return osc, v

        def done(self) -> bool:
            return (self.next and self.dest.midinote == -1) or (
                not self.next and self.midinote == -1
            )

    def __init__(self, **kwargs):
        self.audio: pyaudio.PyAudio = pyaudio.PyAudio()
        self.midinotes: List[Optional[int]] = [None] * 127
        # self.crush = 1
        self.crush = kwargs.get("crush", 1)
        self.func = kwargs.get("func", Wave.SQUARE)
        self.polyphony = kwargs.get("polyphony", 1)
        self.oscs_working = 0  # running total
        self.oscs = [
            Synth.Oscillator(self, self.func, self.crush) for x in range(self.polyphony)
        ]
        # self.buf = array('h', list(range(Synth.FRAMES)))
        self.buf: np.array = np.zeros([Synth.FRAMES], dtype=np.short)
        self.stream: pyaudio.Stream = self.audio.open(
            format=self.audio.get_format_from_width(Synth.WIDTH),
            channels=Synth.CHANNELS,
            rate=Synth.RATE,
            frames_per_buffer=Synth.FRAMES,
            output=True,
            stream_callback=self.callback(),
        )
        self.stream.start_stream()
        self.fxrack = []

    @staticmethod
    def midi_to_pitch(f: float) -> float:
        return pow(2.0, (f - 69.0) / 12.0) * 440.0

    # def callback(self) -> AudioCallback:
    def callback(self):
        def internal_callback(
            in_data: None, frame_count: int, time_info: dict, status: int
        ):
            # for n in range(frame_count):
            #     self.buf[n] = 0
            self.oscs_working = 0
            for o in range(len(self.oscs)):
                osc = self.oscs[o]
                if osc.done():
                    break
                self.oscs_working += 1
                vs = 0
                for n in range(frame_count):
                    # for n in range(frame_count // osc.rate_crush):
                    osc, smp = osc.sample(n)
                    self.oscs[o] = osc
                    v = int(0x7FFF * self.fx(smp))
                    # for i in range(osc.rate_crush):
                    #     idx = n * osc.rate_crush + i
                    #     self.buf[idx] += v
                    self.buf[n] += v
                    # self.buf[idx] = (self.buf[idx] + v) % 0x7fff
            # return (bytes(self.buf), pyaudio.paContinue)
            return (self.buf.tobytes(), pyaudio.paContinue)

        return internal_callback

    # def run(self):
    # self.stream.start_stream()
    # while self.stream.is_active():
    #     time.sleep(0.1)
    def fx(self, v: float) -> float:
        for fxfunc in self.fxrack:
            v = fxfunc(v)
        return v

    def deinit(self):
        for osc in self.oscs:
            if osc.midinote >= 0:
                osc.off()
                while self.oscs_working > 0:
                    time.sleep(0.1)
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def note(self, n: int, v: float = 1.0, ch: int = 0, func=Wave.SINE) -> int:
        o: int = 0
        for osc in self.oscs:
            if osc.done():
                osc = self.oscs[o]
                osc.note(n, v, func=func)
                return o
            o += 1
        return -1
        # osc = self.oscs[0]
        # osc.note(n, v, func=func)
        # return 0

    def off(self, ch: Optional[int] = 0):
        for osc in self.oscs:
            if (ch == None) or ch == osc.ch:
                if not osc.done():
                    osc.off()
        self.oscs.sort(key=lambda x: x.midinote == -1)


def main():
    notelen = 0.29
    notespace = 0.01
    C = 60
    F = 65
    func = Wave.SINE

    synth = Synth()
    synth.note(60, func=Wave.SINE)
    time.sleep(notelen)
    synth.off()
    # time.sleep(notelen)
    # for i in range(7):
    #     synth.note(C + i * 2 - (1 if i >= 3 else 0), func=func)
    #     time.sleep(notelen)
    #     synth.off()
    #     time.sleep(notespace)
    # synth.note(C + 12, func=func)
    # time.sleep(notelen)
    # synth.off()
    # time.sleep(notespace)

    synth.deinit()

    del synth


if __name__ == "__main__":
    main()
