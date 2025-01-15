#######################
# Name: Michael Rideout
# Stduent Id: 225065259
#######################

##################################################################
# Assessment 3 Question 2 Implementation
##################################################################

library(lpSolveAPI)

# Create the linear programming model with 9 variables and 12 constraints
lpmodel <- make.lp(12, 9)

# Set the object to maximize
lp.control(lpmodel, sense= "maximize")

# Set the coefficients for the objective function
set.objfn(lpmodel, c(25, 22, 27, 10, 7, 12, 5, 2, 7)) # set objective function

# Set the demand constraints
set.row(lpmodel, 1, c(1,1,1), indices = c(1,4,7))
set.row(lpmodel, 2, c(1,1,1), indices = c(2,5,8))
set.row(lpmodel, 3, c(1,1,1), indices = c(3,6,9))

# Minimum cotton constraints
set.row(lpmodel, 4, c(0.45,-0.55,-0.55), indices = c(1,4,7))
set.row(lpmodel, 5, c(0.55,-0.45,-0.45), indices = c(2,5,8))
set.row(lpmodel, 6, c(0.7,-0.3,-0.3), indices = c(3,6,9))

# Minimum wool constraints 
set.row(lpmodel, 7, c(-0.3,0.7,-0.3), indices = c(1,4,7))
set.row(lpmodel, 8, c(-0.4,0.6,-0.4), indices = c(2,5,8))
set.row(lpmodel, 9, c(-0.5,0.5,-0.5), indices = c(3,6,9))

# Minimum silk constraints
set.row(lpmodel, 10, c(-0.01,-0.01,0.99), indices = c(1,4,7))
set.row(lpmodel, 11, c(-0.02,-0.02,0.98), indices = c(2,5,8))
set.row(lpmodel, 12, c(-0.03,-0.03,0.97), indices = c(3,6,9))

# Set the RHS for the constraints
set.rhs(lpmodel, c(3200, 3800, 4200, 0, 0, 0, 0, 0, 0, 0, 0, 0))

# Set constraint types
set.constr.type(lpmodel,
                c("<=", "<=", "<=", ">=", ">=", ">=", ">=", ">=", ">=", ">=", ">=", ">="))

# Set data type for decision variables
set.type(lpmodel, c(1:9), "real")

# Set boundary conditions
set.bounds(lpmodel, lower = rep(0, 9), upper = rep(Inf, 9))

# Solve the LP model
solve(lpmodel)

profit <- get.objective(lpmodel)
profit

solution <- get.variables(lpmodel)
solution 

spring <- sum(solution[c(1,4,7)])
autumn <- sum(solution[c(2,5,8)])
winter <- sum(solution[c(3,6,9)])
print("----------------------------------------------------------")
print("Assessement 3 Question 2 Result")
print(paste("Maximised profit: ", profit))
print(paste("Spring tonnage: ", spring, " Autum tonnage: ", autumn, " Winter tonnage: ", winter))
print("----------------------------------------------------------")


##################################################################
# Assessment 3 Question 3 Implementation
##################################################################

#############################
# Player 1 Game 
#############################

library(lpSolveAPI)

# Make the game model
lp_game_player_one <- make.lp(0, 7)

# Set the objective to maximize
lp.control(lp_game_player_one, sense="maximize")

# Set the objective function
set.objfn(lp_game_player_one, c(0, 0, 0, 0, 0, 0, 1))

# Add the constraints for player 1
add.constraint(lp_game_player_one, c(0, 0, 0, -75, 75, 0, 1), "<=", 0)
add.constraint(lp_game_player_one, c(0, 0, -75, 0, 0, 75, 1), "<=", 0)
add.constraint(lp_game_player_one, c(0, 75, 0, 0, 0, -75, 1), "<=", 0)
add.constraint(lp_game_player_one, c(75, 0, 0, 0, -75, 0, 1), "<=", 0)
add.constraint(lp_game_player_one, c(-75, 0, 0, 75, 0, 0, 1), "<=", 0)
add.constraint(lp_game_player_one, c(0, -75, 75, 0, 0, 0, 1), "<=", 0)
add.constraint(lp_game_player_one, c(1, 1, 1, 1, 1, 1, 0), "=", 1)

# Set boundary conditions
set.bounds(lp_game_player_one, lower = c(0, 0, 0, 0, 0, 0, -Inf))

RowNames <- c("Row1", "Row2", "Row3", "Row4", "Row5", "Row6", "Row 7")
ColNames <- c("B1", "B2", "B3", "B4", "B5", "B6", "V")

# Annotate names for rows and columns
dimnames(lp_game_player_one) <- list(RowNames, ColNames)

print("----------------------------------------------------------")
print("Assessement 3 Question 3 Player One Game")

# Solve the LP model
solve(lp_game_player_one)

get.objective(lp_game_player_one)

get.variables(lp_game_player_one)

get.constraints(lp_game_player_one)


#############################
# Player 2 Game 
#############################

library(lpSolveAPI)

# Make the game model
lp_game_player_two <- make.lp(0, 7)

# Set the objective to maximize
lp.control(lp_game_player_two, sense="maximize")

# Set the objective function
set.objfn(lp_game_player_two, c(0, 0, 0, 0, 0, 0, 1))

# Add the constraints for player 1
add.constraint(lp_game_player_two, c(0, 0, 0, 75, -75, 0, 1), "<=", 0)
add.constraint(lp_game_player_two, c(0, 0, 75, 0, 0, -75, 1), "<=", 0)
add.constraint(lp_game_player_two, c(0, -75, 0, 0, 0, 75, 1), "<=", 0)
add.constraint(lp_game_player_two, c(-75, 0, 0, 0, 75, 0, 1), "<=", 0)
add.constraint(lp_game_player_two, c(75, 0, 0, -75, 0, 0, 1), "<=", 0)
add.constraint(lp_game_player_two, c(0, 75, -75, 0, 0, 0, 1), "<=", 0)
add.constraint(lp_game_player_two, c(1, 1, 1, 1, 1, 1, 0), "=", 1)

# Set boundary conditions
set.bounds(lp_game_player_two, lower = c(0, 0, 0, 0, 0, 0, -Inf))

RowNames <- c("Row1", "Row2", "Row3", "Row4", "Row5", "Row6", "Row 7")
ColNames <- c("A1", "A2", "A3", "A4", "A5", "A6", "V")

# Annotate names for rows and columns
dimnames(lp_game_player_two) <- list(RowNames, ColNames)

print("----------------------------------------------------------")
print("Assessement 3 Question 3 Player Two Game")

# Solve the LP model
solve(lp_game_player_two)

get.objective(lp_game_player_two)

get.variables(lp_game_player_two)

get.constraints(lp_game_player_two)