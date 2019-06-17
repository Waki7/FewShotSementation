import numpy as np
from enum import Enum


class DocGenerator():
    def __initd__(self):
        self.heights = [1920]
        self.widths = [1080]
        self.seed = 5
        np.random.seed(5)
        self.horizontalLines = self.createHorizLines()
        self.verticalLines = self.createVertLines()
        self.bucketIndex = 0

    def createHorizLines(self):
        maxWidth = self.widths[self.bucketIndex]
        maxHeight = self.heights[self.bucketIndex]

        x = np.random()

        line = np.random
        pass

    def createVertLines(self):

        pass


def main():
    pass

if __name__ == '__main__':
    main()