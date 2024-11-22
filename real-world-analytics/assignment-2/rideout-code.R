library(pastecs) # For descriptive statistics (for stat.desc)
library(car) # For qqPlot
library(ggplot2) # ggplot
library(knitr) # matrix printing

# Set seed as student id
set.seed(225065259)

################################################################################
# T1 (i) - The dataset was manually downloaded as a zip and extracted from the 
#          following URL - https://d2l.deakin.edu.au/d2l/le/content/1422549/viewContent/7641529/View 
################################################################################

################################################################################
# T1 (ii) - Assign the data to a matrix
################################################################################

# Load the dataset into a matrix, but skip the first column with [, -1] as
# the first column only represents the row number
the.data <- as.matrix(read.table("ENB.txt")[, -1])

# Check that the data is of the expected size
expected_row_count = 19735
expected_column_count = 6
stopifnot(nrow(the.data) == expected_row_count)
stopifnot(ncol(the.data) == expected_column_count)

# Rename the column names
colnames(the.data) <- c("X1", "X2", "X3", "X4", "X5", "Y")

################################################################################
# T1 (iii) - Generate dataset subset
################################################################################

num_row <- 450
num_col <- 6
num_samples <- nrow(the.data)

my.data <- the.data[sample(1:num_samples, num_row), c(1:num_col)]

# assert that we have no missing values
stopifnot(is.na(my.data) == FALSE)

################################################################################
# T1 (iv) - Generate scatter plots and histograms
################################################################################

# Create better alias labels for the columns
living_room_temperature = "Living Room Temperature (celsius) (X1)"
living_room_humidity = "Living Room Humidity (%) (X2)"
office_room_temperature = "Office Room Temperature (celsius) (X3)"
office_room_humidity = "Office Room Humidity (%) (X4)"
pressure = "Pressure (mm of mercury) (X5)"
target = "Appliances Energy Consumption (Wh) (Y)"

# Create column indicies
lr_temp_col_name = "X1"
lr_humidity_col_name = "X2"
or_temp_col_name = "X3"
or_humidity_index = "X4"
pressure_col_name = "X5"
target_col_name = "Y"

# A helper function to create a scatter plot for two variables
scatter_plot <- function(x_data, 
                         y_data,
                         x_label,
                         y_label,
                         title) {
  plot(
    x = x_data,
    y = y_data,
    xlab = x_label,
    ylab = y_label,
    main = title,
    col = "blue"
  )
}

# Generate scatter plots for all X variables against Y
scatter_plot(my.data[,lr_temp_col_name], my.data[,target_col_name], living_room_temperature, target, paste(living_room_temperature, " vs ", target))
scatter_plot(my.data[,lr_humidity_col_name], my.data[,target_col_name], living_room_humidity, target, paste(living_room_humidity, " vs ", target))
scatter_plot(my.data[,or_temp_col_name], my.data[,target_col_name], office_room_temperature, target, paste(office_room_temperature, " vs ", target))
scatter_plot(my.data[,or_humidity_index], my.data[,target_col_name], office_room_humidity, target, paste(office_room_humidity, " vs ", target))
scatter_plot(my.data[,pressure_col_name], my.data[,target_col_name], pressure, target, paste(pressure, " vs ", target))

# A helper function to create a histogram for a variable
histogram <- function(data,   # The data matrix
                      xlabel, # X-axis label
                      title   # Title of the graph
                      ) {
  rounding = 2
  mean <- round(mean(data), rounding)
  median <- round(median(data), rounding)
  min <- round(min(data), rounding)
  max <- round(max(data), rounding)
  
  hist(data, xlab = xlabel, main=title, col="lightblue")
}

generate_statistical_checks <- function(data,           # The data matrix
                                        argument_matrix # matrix to hold args
                            ) {
  for (i in 1:nrow(argument_matrix)) {
    row <- argument_matrix[i, ]
    col_name = row[1]
    label = row[2]
    data_process_level = row[3]
    
    # Generate histograms
    histogram(data[,col_name], label, paste("Historgram of", data_process_level, label))
    
    # QQ Plot
    print(paste("Col name: ", col_name))
    df <- as.data.frame(data)
    plot <-ggplot(df, aes(sample = !!sym(col_name)))+stat_qq()+stat_qq_line(color = "red")+theme_minimal() +                # Apply a minimal theme
      labs(title = paste("Q-Q Plot of", data_process_level, label),
           x = "Theoretical Quantiles",
           y = "Sample Quantiles")
    print(plot)
    
    # Generate descriptive stats
    desc <- stat.desc(data[,col_name], basic=FALSE, norm = TRUE, )
    desc <- round(desc,2)
    print(paste("Descriptive statistics for: ", label))
    print("----------------------------")
    kable(desc)
  }
}

# print pre transform raw graphs of variables
pre_transform_args <- matrix(
  c(lr_temp_col_name, living_room_temperature, " Raw ",
    lr_humidity_col_name, living_room_humidity, " Raw ",
    or_temp_col_name, office_room_temperature, " Raw ",
    or_humidity_index, office_room_humidity, " Raw ",
    pressure_col_name, pressure, " Raw ",
    target_col_name, target, " Raw "
    ),
  ncol=3, byrow = TRUE
)

generate_statistical_checks(my.data, pre_transform_args)



