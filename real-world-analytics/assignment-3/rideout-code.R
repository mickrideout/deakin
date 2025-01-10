#######################
# Name: Michael Rideout
# Stduent Id: 225065259
#######################


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


objvalue <- get.objective(lpmodel)
objvalue

solution <- get.variables(lpmodel)
solution 

spring <- sum(solution[c(1,4,7)])
autumn <- sum(solution[c(2,5,8)])
winter <- sum(solution[c(3,6,9)])
print(paste("Spring tonnage: ", spring, " Autum tonnage: ", autumn, " Winter tonnage: ", winter))

