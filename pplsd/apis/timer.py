# -*- encoding: utf-8 -*-
"""
@File    :   timer.py
@Time    :   2023/12/16 23:15:51
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""


import logging
import time


class Timer:
    """
    Timer
    """

    def __init__(self, cur_iter: int = 0, iters: int = 0):
        self.iters = iters
        self.cur_iter = cur_iter
        self.begin_iter = cur_iter
        self.elasped_time = 0
        self.last_time = None
        self.total_samples = 0

    def step(self):
        """step"""
        self.cur_iter += 1
        now = time.time()

        if self.last_time is not None:
            self.elasped_time += now - self.last_time

        self.last_time = now

    def eta(self):
        """eta"""
        if self.iters == 0:
            return "--:--:--"

        remaining_iter = max(self.iters - self.cur_iter, 0)
        remaining_time = int(
            remaining_iter
            * self.elasped_time
            / (self.cur_iter - self.begin_iter)
        )
        result = "{:0>2}:{:0>2}:{:0>2}"
        arr = []

        for i in range(2, -1, -1):
            arr.append(int(remaining_time / 60**i))
            remaining_time %= 60**i

        return result.format(*arr)


class TimerDecorator:
    """
    Timer decorator
    """

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            """wrapper"""
            T1 = time.time()
            ret = func(*args, **kwargs)
            T2 = time.time()
            logging.info(
                "{} time consuming: {}".format(func.__name__, T2 - T1)
            )
            return ret

        wrapper.__name__ = func.__name__
        return wrapper
