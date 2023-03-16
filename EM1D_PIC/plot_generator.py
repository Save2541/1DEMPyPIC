from . import plot_config


class PlotList:
    def __init__(self, shape=None, keys=None, overlays=None):
        """
        Generate a list of plots
        :param shape: shape of the list
        :param keys: values to be plotted
        :param overlays: keys of the lines to be overlaid on the plots
        """
        self.shape = shape
        self.keys = keys
        self.overlays = overlays

        # SET DEFAULT VALUES
        if shape is None:
            self.shape = plot_config.plots
        if keys is None:
            self.keys = plot_config.plots_key
        if overlays is None:
            self.overlays = plot_config.overlays

    def title(self, row_id, column_id):
        """
        Return the title of a plot
        :param row_id: row index
        :param column_id: column index
        :return: plot title
        """
        return self.keys[row_id][column_id].capitalize()

    def row_range(self):
        """
        Return the range of row indices
        :return: range of row indices
        """
        return range(self.shape[0])

    def column_range(self):
        """
        Return the range of column indices
        :return: range of column indices
        """
        return range(self.shape[1])


