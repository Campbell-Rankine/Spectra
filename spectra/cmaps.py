import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Colormap
import matplotlib.pyplot as plt


class LinearGradient(LinearSegmentedColormap):
    def __init__(self, name: str, colour_format: str, colours: list):
        self.name = name
        self.colours = self.create_colours(colours, colour_format)

    @property
    def cmap(self):
        return LinearSegmentedColormap.from_list(self.name, self.colours)

    def create_colours(self, colours, colour_format: str):
        match colour_format:
            case "rgb":
                return self._from_rgb(colours)
            case "hex":
                return self._from_hex(colours)

    def _from_rgb(self, colours):
        if not isinstance(colours, np.ndarray):
            colours = np.array(colours)
        colours = colours[:, :3] / 255
        return colours

    def _from_hex(self, colours):
        if not isinstance(colours, np.ndarray):
            colours = np.array(colours)
        return colours

    def get_inverted(self):
        inverted_colours = 1.0 - np.asarray(self.colours)[:, :3]
        return LinearSegmentedColormap.from_list(
            f"{self.name}-inverted", inverted_colours
        )


spectra = LinearGradient(
    "spectra",
    "hex",
    [
        "#1A4C61",
        "#41915B",
        "#D94332",
    ],
)

spectra_warm = LinearGradient(
    "spectra_warm",
    "hex",
    [
        "#1C5AA0",
        "#D1003F",
        "#FF572D",
    ],
)


# spectra_inverted = spectra.get_inverted()
