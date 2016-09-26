#!/usr/bin/env python
# -*- coding=utf-8 -*-

art_dict = {}

def getArtLines(ascii):
    lineNumberStart = (ascii - 64 - 1)*3 + 1
    result = []
    lines = ['┌─┐', '├─┤', '┴ ┴', '┬┐ ', '├┴┐', '┴─┘', '┌─┐', '│  ', '└─┘', '┬─┐', '│ │', '┴─┘', '┌─┐', '├┤ ', '└─┘', '┬─┐', '├┤ ', '┴  ', '┌─┐', '│ ┐', '└─┘', '┬ ┬', '├─┤', '┴ ┴', ' ┬ ', ' │ ', ' ┴ ', ' ┬ ', ' │ ', '└┘ ', '┐┌ ', '├┴┐', '┴ ┴', '┬  ', '│  ', '┴─┘', '┌┬┐', '│││', '┴ ┴', '┌┐┌', '│││', '┘└┘', '┌─┐', '│ │', '└─┘', '┬─┐', '├─┘', '┴  ', '┌─┐', '│ │', '└─┴', '┬─┐', '├┬┘', '┴└─', '┌─┐', '└─┐', '└─┘', '┌┬┐', ' │ ', ' ┴ ', '┐ ┌', '│ │', '└─┘', '┐ ┌', '└┐│', ' └┘', '┬ ┬', '│││', '└┴┘', '┬ ┬', '└┬┘', '┴ ┴', '┬ ┬', '└─┤', '└─┘', '┌─┐', '┌─┘', '└─┘']
    for i in range(3):
        result.append(lines[lineNumberStart+i-1])
    return result

# Init dictionary to store ascii art
for ascii in range(65,91):
    char = chr(ascii)
    art_dict[char] = getArtLines(ascii)
art_dict[" "] = ['   ', '   ', '   '] # for space

str = raw_input("Type what you want to generate character ascii art:").upper()

artLines = ["", "", ""]

# parse user input
for s in str:
    # combine each lines
    artLines[0] += (art_dict[s][0])
    artLines[1] += (art_dict[s][1])
    artLines[2] += (art_dict[s][2])

# echo combined each line
for i in range(3):
    print(artLines[i])
