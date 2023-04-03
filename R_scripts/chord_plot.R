setwd("/home/hujin/jin/MyResearch/gTVDN/")
# pkg for chord plot
library(circlize)
# pkg for saving jpg in linux without X11
library(Cairo)

# input a matrix manually or load 
# data_matrix <- matrix(c(
#   24,  6,  4, 10,  6, 17, 12,
#   8, 23,  3,  8,  0, 17,  8,
#   8,  4, 20,  7,  3, 19,  9,
#   9,  9, 13, 24,  4, 16, 13,
#   5,  3,  2,  4,  9,  5, 10,
#   20, 19, 21, 20,  6, 89, 18,
#   9,  5,  7, 18,  6, 25, 41), 
#   nrow = 7, byrow = TRUE)

# the title name of the plot
title.name <- "AD"
# file name of the txt file, the saved jpg file is in the same name
#file.name <- "chord_plot_U_AD"
file.name <- "chord_plot_kmeans_6_AD"
data_matrix <- as.matrix(read.table(paste0("./mid_results/", file.name, ".txt"), sep=" "))


num.state <- nrow(data_matrix)
row.names(data_matrix) <- colnames(data_matrix) <- paste0("S-", 1:num.state)
#row.names(data_matrix) <- paste0("State ", 1:num.state)

# remove small links
non_diag_eles <- data_matrix[!diag(num.state)]
cut.off <- median(non_diag_eles)

# set for grid colors
set.seed(1)
grid.cols <- rand_color(num.state, transparency=0.5)
names(grid.cols) <- paste0("State ", 1:num.state)


# Set the output file type, file name, and resolution
#jpeg(paste0("./figs/", file.name, ".jpg"), width = 1000, height = 1000, res = 300)
CairoJPEG(paste0("./figs/", file.name, ".jpg"), width = 1000, height = 1000, res = 300)
# remove the margin of the plot
par(mar = c(0, 0, 1, 0))

# rotate for 90 degrees
circos.par(start.degree = 90)
# link.visible: to remove the small link between states
chordDiagram(data_matrix, 
             grid.col = grid.cols,
             link.visible = data_matrix>cut.off)
title(title.name)

# Close the graphics device to save the plot to the file
dev.off()
circos.clear()

