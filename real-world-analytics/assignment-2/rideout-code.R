library(pastecs) # For descriptive statistics (for stat.desc)
library(car) # For qqPlot
library(ggplot2) # ggplot
library(knitr) # matrix printing
library(corrplot) # correlation matrix plotting

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

# Filter out outliers from the dataset before sampling
remove_outliers <- function(data) {
  valid_rows <- rep(TRUE, nrow(data))
  
  for (col in 1:ncol(data)) {
    q1 <- quantile(data[, col], 0.25)
    q3 <- quantile(data[, col], 0.75)
    iqr <- q3 - q1
    
    lower_bound <- q1 - 1.5 * iqr
    upper_bound <- q3 + 1.5 * iqr
    
    valid_rows <- valid_rows & (data[, col] >= lower_bound & data[, col] <= upper_bound)
  }
  return (data[valid_rows, ])
}

filtered_data <- remove_outliers(the.data)
num_samples <- nrow(filtered_data)

# Sample the data to have 450 rows
my.data <- filtered_data[sample(1:num_samples, num_row), c(1:num_col)]

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

# Create column indices
lr_temp_col_name = "X1"
lr_humidity_col_name = "X2"
or_temp_col_name = "X3"
or_humidity_col_name = "X4"
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
scatter_plot(my.data[,lr_temp_col_name],
             my.data[,target_col_name],
             living_room_temperature,
             target, 
             paste(living_room_temperature, " vs ", target))
scatter_plot(my.data[,lr_humidity_col_name],
             my.data[,target_col_name],
             living_room_humidity,
             target,
             paste(living_room_humidity, " vs ", target))
scatter_plot(my.data[,or_temp_col_name],
             my.data[,target_col_name],
             office_room_temperature,
             target,
             paste(office_room_temperature, " vs ", target))
scatter_plot(my.data[,or_humidity_col_name],
             my.data[,target_col_name],
             office_room_humidity,
             target,
             paste(office_room_humidity, " vs ", target))
scatter_plot(my.data[,pressure_col_name],
             my.data[,target_col_name],
             pressure,
             target,
             paste(pressure, " vs ", target))

# A helper function to create a histogram for a variable
histogram <- function(data,   # The data matrix
                      xlabel, # X-axis label
                      title   # Title of the graph
                      ) {
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
    plot <-ggplot(df, aes(sample = !!sym(col_name)))+stat_qq()+stat_qq_line(color = "red")+theme_minimal()+labs(title = paste("Q-Q Plot of", data_process_level, label),
           x = "Theoretical Quantiles",
           y = "Sample Quantiles")
    print(plot)
    
    # Generate descriptive stats
    desc <- stat.desc(data[,col_name], basic=FALSE, norm = TRUE, )
    desc <- round(desc,2)
    print(paste("Descriptive statistics for: ", label))
    print("----------------------------")
    print(kable(desc))
    print("----------------------------")
  }
}

# print pre transformed raw graphs of variables
pre_transform_args <- matrix(
  c(lr_temp_col_name, living_room_temperature, " Raw ",
    lr_humidity_col_name, living_room_humidity, " Raw ",
    or_temp_col_name, office_room_temperature, " Raw ",
    or_humidity_col_name, office_room_humidity, " Raw ",
    pressure_col_name, pressure, " Raw ",
    target_col_name, target, " Raw "
    ),
  ncol=3, byrow = TRUE
)

generate_statistical_checks(my.data, pre_transform_args)

# Generate the correlation plot
cor_matrix <- cor(my.data) 
corrplot(cor_matrix, method = "circle", addCoef.col = 'black')


################################################################################
# T2 - Transform the data
################################################################################

# Drop column we are not interested in
drop_col_name = lr_temp_col_name
my.data <- my.data[, colnames(my.data) != drop_col_name]
stopifnot(ncol(my.data) == 5) # assert that we now have 5 columns

# Generate statistical descriptives of mean and standard variation for 
# our 4 features and 1 target for use in zscore calculation
# and also for inference
generate_std_and_mean <- function(data, # the data matrix
                                  col_names # array of column names
) {
  result_list <- list()
  for (col_name in col_names) {
    stats <- list(
      std = sd(data[, col_name]),
      mean = mean(data[, col_name])
    )
    result_list[[col_name]] = stats
  }
  result_list
}

# Calculate the zscore for a particular column
unit_zscore <- function(data,
                        col_name,
                        std_and_means) {
  0.15 * ((data - std_and_means[[col_name]][["mean"]]) / std_and_means[[col_name]][["std"]]) + 0.5
}


# Function to transform the matrix data
# arg: data - a data matrix with 5 columns (the last being the target)
# arg: apply_to_target - apply transform target or not
apply_distribution_transforms <- function(data,
                                          apply_to_target=TRUE) {
  #data[, lr_temp_col_name] <- 1 / (data[, lr_temp_col_name])
  data[, or_humidity_col_name] <- 1 / (data[, or_humidity_col_name] + 0.001) # constant added to avoid division by zero
  data[, pressure_col_name] <- (data[, pressure_col_name])^3.5
  if (apply_to_target) {
    data[, target_col_name] <- log(data[, target_col_name])
  }
  data
  
}

# Apply unit zscore transformation to all columns / target
# arg: data - a matrix containing the dataset
# arg: std_and_means - a list of lists of the standard deviation and mean for 
#  the dataset for each  feature / target
# arg: apply_to_target - apply transform target or not
apply_unit_zscore_transforms <- function(data,
                                         std_and_means,
                                         apply_to_target=TRUE) {
  #data[, lr_temp_col_name] <- unit_zscore(data[, lr_temp_col_name], lr_temp_col_name, std_and_means)
  data[, lr_humidity_col_name] <- unit_zscore(data[, lr_humidity_col_name], lr_humidity_col_name, std_and_means)
  data[, or_temp_col_name] <- unit_zscore(data[, or_temp_col_name], or_temp_col_name, std_and_means)
  data[, or_humidity_col_name] <- unit_zscore(data[, or_humidity_col_name], or_humidity_col_name, std_and_means)
  data[, pressure_col_name] <- unit_zscore(data[, pressure_col_name], pressure_col_name, std_and_means)
  if (apply_to_target) {
    data[, target_col_name] <- unit_zscore(data[, target_col_name], target_col_name, std_and_means)
  }
  data
}

# Transform the data
transformed.data <- apply_distribution_transforms(my.data)
std_and_means <- generate_std_and_mean(transformed.data, 
                                       c(
                                         #lr_temp_col_name,
                                         lr_humidity_col_name,
                                         or_temp_col_name,
                                         or_humidity_col_name,
                                         pressure_col_name,
                                         target_col_name)) 
transformed.data <- apply_unit_zscore_transforms(transformed.data, std_and_means)



# graph transformed data
# print pre transformed raw graphs of variables
post_transform_args <- matrix(
  c(
    #lr_temp_col_name, living_room_temperature, " Transformed ",
    lr_humidity_col_name, living_room_humidity, " Transformed ",
    or_temp_col_name, office_room_temperature, " Transformed ",
    or_humidity_col_name, office_room_humidity, " Transformed ",
    pressure_col_name, pressure, " Transformed ",
    target_col_name, target, " Transformed "
  ),
  ncol=3, byrow = TRUE
)

generate_statistical_checks(transformed.data, post_transform_args)

# Save the transformed data 
write.table(transformed.data, "name-transformed.txt")

################################################################################
# T3 - Model Construction
################################################################################

# (i) Load library
source("AggWaFit718.R")

# (ii) a - Weighted Arithmetic Mean
fit.QAM(
  transformed.data,
  output.1 = "wam_results.txt",
  stats.1 = "wam_statistics.txt"
)

# (ii) b - Weighted power means using p = 0.5
fit.QAM(
  transformed.data,
  output.1 = "wpm_p0.5_results.txt",
  stats.1 = "wpm_p0.5_statistics.txt",
  g = PM05,
  g.inv = invPM05
)

## (ii) c - Weighted power means using p = 2
fit.QAM(
  transformed.data,
  output.1 = "wpm_p2_results.txt",
  stats.1 = "wpm_p2_statistics.txt",
  g = QM,
  g.inv = invQM
)

## (ii) d - Ordered Weighted Averaging function
fit.OWA(
  transformed.data,
  output.1 = "owa_results.txt",
  stats.1 = "owa_statistics.txt"
)

################################################################################
# T4 - Model Prediction
################################################################################

# Manually define the matrix from the data in the assignment specification sheet
inference_matrix <- matrix(
  c(19.1, 43.29, 19.7, 43.4, 743.6),
  ncol=5, byrow = TRUE
)
colnames(inference_matrix) <- c(lr_temp_col_name,
                                lr_humidity_col_name,
                                or_temp_col_name,
                                or_humidity_col_name,
                                pressure_col_name)

print(inference_matrix)
print(is.matrix(inference_matrix))


# Drop the column we have decided not to use
inference_matrix <- inference_matrix[, colnames(inference_matrix) != drop_col_name, drop=FALSE]
stopifnot(ncol(inference_matrix) == 4) # assert that we now have 4 columns

# Transform the inference_matrix using the same transformations as the training data
inference_matrix <- apply_distribution_transforms(inference_matrix, apply_to_target=FALSE)
inference_matrix <- apply_unit_zscore_transforms(inference_matrix, std_and_means, apply_to_target=FALSE)

# Define the weights learnt via training of the Weighted Average Model
weights <- c(0.145113752792045,
             0.351562163999998,
             0.349343633342744,
             0.153980449865216)

# Use the Weighted Average Model to make the prediction
target_prediction_scaled <- QAM(inference_matrix, weights)

print(target_prediction_scaled)

# We have a target prediction but it is log transformed and scaled. We need to reverse this
target_prediciton = exp((std_and_means[[target_col_name]][["std"]] * (target_prediction_scaled - 0.5) / 0.15) 
                        + std_and_means[[target_col_name]][["mean"]])

print(target_prediciton)
