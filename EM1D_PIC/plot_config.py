plot_type = 2  # 1: Fourier spectrum plot, 2: phase space animation, 3: phase space plot at a specific time

# FOURIER SPECTRUM PLOT CONFIG (plot_type = 1)

# PHASE SPACE CONFIG (plot_type = 2 AND 3)
add_plots = (2, 4)  # set space for additional plots which will appear above the phase space animation (rows, columns)
add_plots_keys = (("rho", "by", "bz", "jy"), ("ex", "ey", "ez", "jz"))  # values to be animated in the additional plots
fps = 120  # frame rate per second
plot_colors = ("black", "red", "yellow")  # color to use for each specie, ordered by specie indices
specie_of_interest = 0  # index of the specie to focus on in the phase space animation (set to 0 to not focus)
dot_size = 1  # particle size in phase space plot
frame_interval = 16  # number of time steps to skip per frame plus one (set to 1 to plot every recorded time step)

# FOR plot_type = 3 only
time = 0  # index of the time step to be plotted
