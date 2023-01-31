# LIST OF FILES TO BE PLOTTED (FILE NAMES WITHOUT EXTENSIONS)
file_list = ["EM_20230131-142331"]

# CHOOSE PLOT TYPE
plot_type = [1]  # 1: Fourier spectrum plot, 2: phase space animation, 3: phase space plot at a specific time

# FOURIER SPECTRUM PLOT CONFIG (plot_type = 1)
plots = (2, 3)  # set space for plots (# of rows, # of columns)
plots_key = (("ex", "ey", "ez"), ("rho", "by", "bz"))  # values to be plotted
electron_id = 0  # index of electrons in the specie list
ion_id = 1  # index of ions in the specie list (if no ion, input None)
waves_of_interest = 1  # (0 for electron waves, 1 for ion waves)
# lines to be overlaid on the plots (use None for no line, "auto" only works for theta = 0 or 90)
overlays = (("auto", "auto", "auto"), ("auto", "auto", "auto"))
orders_of_magnitude = 4  # orders of magnitude in the color bar
maximum_magnitude = 0  # use to lower the maximum order of magnitude in the color bar
plot_range_x = (0, 2)  # x-axis plot range (input: "full" for full plot)
plot_range_y = (0, 1E7)  # y-axis plot range (input: "full" for full plot)

# PHASE SPACE CONFIG (plot_type = 2 or 3)
add_plots = (2, 4)  # set space for additional plots which will appear above the phase space animation (rows, columns)
add_plots_keys = (("rho", "by", "bz", "jy"), ("ex", "ey", "ez", "jz"))  # values to be animated in the additional plots
fps = 120  # frame rate per second
plot_colors = ("black", "red", "yellow")  # color to use for each specie, ordered by specie indices
specie_of_interest = 0  # index of the specie to focus on in the phase space animation (set to 0 to not focus)
dot_size = 1  # particle size in phase space plot
frame_interval = 16  # number of time steps to skip per frame plus one (set to 1 to plot every recorded time step)

# FOR plot_type = 3 only
time = 0  # index of the time step to be plotted
