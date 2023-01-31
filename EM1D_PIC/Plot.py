import io

from EM1D_PIC import animation
from EM1D_PIC import spectrum_plot
from EM1D_PIC import plot_config


def run_plot(file_list=None, plot_type=None):
    """
    Runs the plotting program
    :param file_list: list of data files
    :param plot_type: plot types
    :return:
    """
    # GET FILE LIST FROM CONFIG
    if file_list is None:
        file_list = plot_config.file_list
    # GET PLOT TYPE FROM CONFIG
    if plot_type is None:
        plot_type = plot_config.plot_type
    # RUN PROGRAM WITH PROFILER
    if __name__ == '__main__':
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.enable()
        for file in file_list:
            for key in plot_type:
                if key == 1:
                    spectrum_plot.plot_fourier(file)
                elif key == 2:
                    animation.plot_non_fourier(file)
                elif key == 3:
                    animation.plot_non_fourier(file, anim=False)
        profiler.disable()
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')
        stats.print_stats()

        with open('outstats_ps.txt', 'w+') as f:
            f.write(s.getvalue())


run_plot()
