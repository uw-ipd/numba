# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
import sys

empty_colors = dict.fromkeys(
    ["pink", "blue", "green", "yellow", "red", "uncolored",], "")

ansi_colors = {
    "pink": '\033[95m',
    "blue": '\033[94m',
    "green": '\033[92m',
    "yellow": '\033[93m',
    "red": '\033[91m',
    "uncolored": '\033[0m',
}


class Colorer(object):
    def __init__(self, colors):
        self.colors = colors

    def color_text(self, text, color_name):
        return "%s%s%s" % (self.colors[color_name],
                           text,
                           self.colors["uncolored"])

empty_colorer = Colorer(empty_colors)
ansi_colorer = Colorer(ansi_colors)

if sys.platform == "win32" or not sys.stdout.isatty():
    colorer = empty_colorer
else:
    colorer = ansi_colorer


def test(colorer):
    print colorer.color_text("hello world", "pink"), "normal"
    print colorer.color_text("hello world", "blue")
    print colorer.color_text("hello world", "green")
    print colorer.color_text("hello world", "yellow")
    print colorer.color_text("hello world", "red")

if __name__ == '__main__':
    test(empty_colorer)
    test(ansi_colorer)
