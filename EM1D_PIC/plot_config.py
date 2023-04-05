# LIST OF FILES TO BE PLOTTED (FILE NAMES WITHOUT EXTENSIONS)
file_list = ["ES_20230405-164627"]

# CHOOSE PLOT TYPE
plot_type = [1]  # 1: Fourier spectrum plot, 2: phase space animation, 3: phase space plot at a specific time

# FOURIER SPECTRUM PLOT CONFIG (plot_type = 1)
plots = (2, 3)  # set space for plots (# of rows, # of columns)
plots_key = (("ex", "ey", "ez"), ("rho", "by", "bz"))  # values to be plotted
electron_id = 0  # index of electrons in the specie list
ion_id = 1  # index of ions in the specie list (if no ion, input None)
waves_of_interest = 1  # (0 for electron waves, 1 for ion waves)
# lines to be overlaid on the plots (use None for no line, "auto" only works for theta = 0 or 90)
overlays = (("auto", None, None), ("auto", None, None))
orders_of_magnitude = 3  # orders of magnitude in the color bar
maximum_magnitude = 1  # use to lower the maximum order of magnitude in the color bar
plot_range_x = (0, 6)  # x-axis plot range (input: "full" for full plot)
plot_range_y = (0, 6)  # y-axis plot range (input: "full" for full plot)

# PHASE SPACE CONFIG (plot_type = 2 or 3)
add_plots = (2, 4)  # set space for additional plots which will appear above the phase space animation (rows, columns)
add_plots_keys = (("rho", "by", "bz", "jy"), ("ex", "ey", "ez", "jz"))  # values to be animated in the additional plots
fps = 144  # frame rate per second
plot_species = (1,)  # species to plot in phase space
plot_colors = ("red",)  # color to use for each specie, same order as plot_species
specie_of_interest = 2  # index of the specie to focus on in the phase space animation (set to None to not focus)
dot_size = 0.1  # particle size in phase space plot
frame_interval = 1  # number of time steps to skip per frame plus one (set to 1 to plot every recorded time step)

# FOR plot_type = 3 only
time = 0  # index of the time step to be plotted
