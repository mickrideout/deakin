#################################################
#############                       #############
#############  AggWAfit             #############
#############                       #############
#################################################

#  The following functions can be used for calculating and fitting aggregation functions to data
#  
#  For fitting, the data table needs to be in the form x_11 x_12 ... x_1n y_1, i.e. with the first
#        n columns representing the variables and the last column representing the output.


###############################################################
# NECESSARY LIBRARIES (will require installation of packages) #
###############################################################

library(lpSolve) # (Csárdi and Berkelaar, 2024)
#library(scatterplot3d)

library(pastecs) # For descriptive statistics (for stat.desc) (Grosjean et al, 2024)
library(car) # dependency for ggplot2 (Fox et al, 2024)
library(ggplot2) # graph plotting library (Wickham et al., 2024)
library(knitr) # matrix printing (Xie et al., 2024)
library(corrplot) # correlation matrix plotting (Wei et al., 2024)
library(MonoInc) # library for testing monotonicity (Minto et al., 2024)

# Set seed as student id
set.seed(225065259)

########################
# FUNCTION DEFINITIONS #
########################

#------ some generators ------#
AM <- function(x) {x}
invAM <- function(x) {x}
GM <- function(x) {-log(x)}
invGM <- function(x) {exp(-x)}
GMa <- function(x)    {x^0.00001}
invGMa <- function(x) {x^(1/0.00001)}
QM <- function(x) {x^2}
invQM <- function(x) {sqrt(x)}
PM05 <- function(x) {x^0.5}
invPM05 <-function(x) {x^(1/0.5)}
HM <- function(x) {x^(-1)}
invHM <- function(x) {x^(-1)}


#------ Weighted Power Means ------#

PM <- function(x,w =array(1/length(x),length(x)),p) {    # 1. pre-defining the function inputs
  if(p == 0) {                  # 2. condition for `if' statement
    prod(x^w) 			          # 3. what to do when (p==0) is TRUE
  }
  else {(sum(w*(x^p)))^(1/p)}   # 4. what to do when (p==0) is FALSE
}



#------ Weighted Quasi-Arithmetic Means ------#

QAM <- function(x,w=array(1/length(x),length(x)),g=AM,g.inv=invAM) { # 1. pre-defining the inputs 
  #    (with equal weights and g ~arithmetic mean default)
  n=length(x)														# 2. store the length of x
  for(i in 1:n) x[i] <- g(x[i])									# 3. transform each of the inputs 
  #    individually in case definition of g can't operate
  #    on vectors
  g.inv(sum(x*w))													# 4. QAM final calculation
}


#------ OWA  ------# 

#    Note that this calculates the OWA assuming the data are arranged from lowest to highest - this is opposite to a number of publications but was used in the text for consistency with other functions requiring a reordering of the inputs.

OWA <- function(x,w=array(1/length(x),length(x)) ) {    # 1. pre-defining the inputs (with equal weights default)
  w <- w/sum(w)											# 2. normalise the vector in case weights don't add to 1
  sum(w*sort(x))							# 3. OWA calculation
}


#------ Choquet Integral  ------#

choquet <- function(x,v) {   						# 1. pre-defining the inputs (no default)
  n <- length(x)               						# 2. store the length of x
  w <- array(0,n)             						# 3. create an empty weight vector
  for(i in 1:(n-1)) {          						# 4. define weights based on order
    v1 <- v[sum(2^(order(x)[i:n]-1))]     			#
    # 4i. v1 is f-measure of set of all 
    #     elements greater or equal to 
    #     i-th smallest input.   
    v2 <- v[sum(2^(order(x)[(i+1):n]-1))] 			#
    # 4ii. v2 is same as v1 except 
    #     without i-th smallest 
    w[i] <-  v1 - v2           						# 4iii. subtract to obtain w[i]  
  }                         						#
  w[n] <- 1- sum(w)            						# 4iv. final weight leftover            
  x <- sort(x)                 						# 5. sort our vector
  sum(w*x)                     						# 6. calculate as we would WAM
}



#############################
# PLOTTING FUNCTIONS #
#############################

#------ 3D mesh plot  ------#

f.plot3d <- function(f,x.dom = c(0,1), y.dom = c(0,1),grid = c(25,25)) {
  all.points <- array(0,0)
  for(j in 0:(grid[2])) {for(i in 0:(2*grid[1]))  {
    all.points <- rbind(all.points,c(x.dom[1]+abs(grid[1]-i)*(x.dom[2]-x.dom[1])/grid[1],y.dom[1]+j*(y.dom[2]-y.dom[1])/grid[2])    )
  }}
  for(j in grid[1]:0) {for(i in 0:(2*grid[2]))  {
    all.points <- rbind(all.points,c(x.dom[1]+j*(x.dom[2]-x.dom[1])/grid[1], y.dom[1]+abs(grid[2]-i)*(y.dom[2]-y.dom[1])/grid[2]    ))
  }
  }
  
  all.points <- cbind(all.points,0)
  for(i in 1:nrow(all.points)) all.points[i,3] <- f(all.points[i,1:2])
  
  scatterplot3d(all.points,type="l",color="red",xlab="y",ylab="",zlab="",angle=150,scale.y=0.5,grid=FALSE,lab=c(3,3),x.ticklabs=c(x.dom[1],(x.dom[2]-x.dom[1])/2+x.dom[1],x.dom[2]),y.ticklabs=c(y.dom[1],(y.dom[2]-y.dom[1])/2+y.dom[1],y.dom[2]))
  text(-0.85,0,"x")
}

#############################
# FITTING FUNCTIONS TO DATA #
#############################


#------ fit.QAM  (finds the weighting vector w and outputs new y-values)  ------#
#
#  This function can be used to find weights for any power mean (including arithmetic means, geometric mean etc.)
#  It requires the generators (defined above), so for fitting power means, the arguments g= and g.inv= can be changed
#     appropriately.
#  It outputs the input table with the predicted y-values appended and a stats file.  To avoid overwriting
#     these files, you will need to change the output name each time.  The stats file includes measures
#     of correlation, RMSE, L1-error and the orness of the weighting vector.
#  The fitting can be implemented on a matrix A using
#     fit.OWA(A,"output.file.txt","output.stats.txt"). 


fit.QAM <- function(the.data,output.1="output1.txt",stats.1="stats1.txt",g=AM,g.inv=invAM) {
  # preliminary information
  ycol <- ncol(the.data)
  n <- ycol-1
  instances <- nrow(the.data)
  
  # build constraints matrix 
  all.const <- array(0,0)
  # reordered g(x_i)
  for(k in 1:instances) {const.i <- as.numeric(the.data[k,1:n]) 
  for(j in 1:n) const.i[j] <- g(const.i[j])
  all.const <- rbind(all.const,const.i)
  }
  # residual coefficients
  resid.pos <- -1*diag(instances)
  resid.neg <- diag(instances)
  # merge data constraints f - rij = y
  all.const <- cbind(all.const,resid.pos,resid.neg)
  # add row for weights sum to 1
  all.const<-rbind(all.const,c(array(1,n),array(0,2*instances)))
  
  # enforce weights >0
  w.geq0 <- diag(n)
  w.geq0 <- cbind(w.geq0,array(0,c(n,2*instances)))
  # add weight constraints to matrix
  all.const<-rbind(all.const,w.geq0)
  # create rhs of constr
  constr.v <- array(0,nrow(all.const))
  for(i in 1:instances) {
    # populate with y observed
    constr.v[i] <- g(the.data[i,ycol])
    # weights sum to 1
    constr.v[instances+1] <- 1
    # remainder should stay 0
  }		
  for(i in (instances+2):length(constr.v)) {constr.v[i] <- 0}								
  # create inequalities direction vector
  constr.d <- c(array("==",(instances+1)),array(">=",n))
  
  # objective function is sum of resids
  obj.coeff <- c(array(0,n),array(1,2*instances))
  # solve the lp to find w
  lp.output<-lp(direction="min",obj.coeff,all.const,constr.d,constr.v)$solution
  # create the weights matrix
  w.weights<-array(lp.output[1:n])
  # calculate predicted values
  new.yvals <- array(0,instances)
  for(k in 1:instances) {
    new.yvals[k] <- QAM(the.data[k,1:n],(w.weights),g,g.inv)
  }
  # write the output
  
  write.table(cbind(the.data,new.yvals),output.1,row.names = FALSE, col.names = FALSE)
  # write some stats	
  RMSE <- (sum((new.yvals - the.data[,ycol])^2)/instances)^0.5
  av.l1error <- sum(abs(new.yvals - the.data[,ycol]))/instances
  
  somestats <- rbind(c("RMSE",RMSE),c("Av. abs error",av.l1error),c("Pearson correlation",cor(the.data[,ycol],new.yvals)),c("Spearman correlation",cor(the.data[,ycol],new.yvals,method="spearman")),c("i","w_i "),cbind(1:n,w.weights))
  
  write.table(somestats,stats.1,quote = FALSE,row.names=FALSE,col.names=FALSE)	
}



#------ fit.OWA  (finds the weighting vector w and outputs new y-values)  ------#
#
#  This function can be used to find weights for the OWA.
#  It outputs the input table with the predicted y-values appended and a stats file.  To avoid overwriting
#     these files, you will need to change the output name each time.  The stats file includes measures
#     of correlation, RMSE, L1-error and the orness of the weighting vector.
#  The fitting can be implemented on a matrix A using
#     fit.OWA(A,"output.file.txt","output.stats.txt"). 

# reads data as x1 ... xn y

fit.OWA <- function(the.data,output.1="output1.txt",stats.1="stats1.txt") {
  # preliminary information										
  ycol <- ncol(the.data)
  n <- ycol-1
  instances <- nrow(the.data)
  
  # build constraints matrix 
  all.const <- array(0,0)
  # reordered g(x_i)
  for(k in 1:instances) {const.i <- as.numeric(sort(the.data[k,1:n])) 
  all.const <- rbind(all.const,const.i)
  }
  # residual coefficients
  resid.pos <- -1*diag(instances)
  resid.neg <- diag(instances)
  # merge data constraints f - rij = y
  all.const <- cbind(all.const,resid.pos,resid.neg)
  # add row for weights sum to 1
  all.const<-rbind(all.const,c(array(1,n),array(0,2*instances)))
  
  # enforce weights >0
  w.geq0 <- diag(n)
  w.geq0 <- cbind(w.geq0,array(0,c(n,2*instances)))
  # add weight constraints to matrix
  all.const<-rbind(all.const,w.geq0)
  # create rhs of constr
  constr.v <- array(0,nrow(all.const))
  for(i in 1:instances) {
    # populate with y observed
    constr.v[i] <- the.data[i,ycol]
    # weights sum to 1
    constr.v[instances+1] <- 1
    # remainder should stay 0
  }		
  for(i in (instances+2):length(constr.v)) {constr.v[i] <- 0}								
  # create inequalities direction vector
  constr.d <- c(array("==",(instances+1)),array(">=",n))
  
  # objective function is sum of resids
  obj.coeff <- c(array(0,n),array(1,2*instances))
  # solve the lp to find w
  lp.output<-lp(direction="min",obj.coeff,all.const,constr.d,constr.v)$solution
  # create the weights matrix
  w.weights<-array(lp.output[1:n])
  # calculate predicted values
  new.yvals <- array(0,instances)
  for(k in 1:instances) {
    new.yvals[k] <- OWA(the.data[k,1:n],t(w.weights))
  }
  # write the output						
  
  
  write.table(cbind(the.data,new.yvals),output.1,row.names = FALSE, col.names = FALSE)
  # write some stats	
  RMSE <- (sum((new.yvals - the.data[,ycol])^2)/instances)^0.5
  av.l1error <- sum(abs(new.yvals - the.data[,ycol]))/instances
  
  
  somestats <- rbind(c("RMSE",RMSE),c("Av. abs error",av.l1error),c("Pearson correlation",cor(the.data[,ycol],new.yvals)),c("Spearman correlation",cor(the.data[,ycol],new.yvals,method="spearman")),c("Orness",sum(w.weights*(1:n-1)/(n-1))), c("i","w_i "),cbind(1:n,w.weights))
  
  write.table(somestats,stats.1,quote = FALSE,row.names=FALSE,col.names=FALSE)	
}


#------ fit.choquet  (finds the weighting vector w and outputs new y-values)  ------#
#
#  This function can be used to find weights for the OWA.
#  It outputs the input table with the predicted y-values appended and a stats file.  To avoid overwriting
#     these files, you will need to change the output name each time.  The stats file includes measures
#     of correlation, RMSE, L1-error and the orness of the weighting vector.
#  The fitting can be implemented on a matrix A using
#     fit.OWA(A,"output.file.txt","output.stats.txt"). 

fit.choquet <- function(the.data,output.1="output1.txt",stats.1="stats1.txt",kadd=(ncol(the.data)-1)) {
  # preliminary information
  ycol <- ncol(the.data)
  n <- ycol - 1
  instances <- nrow(the.data)
  numvars <- 1
  for(i in 1:kadd) {numvars <- numvars + factorial(n)/(factorial(i)*factorial(n-i))}
  # build cardinality data sets
  card <- rbind(0,t(t(1:n)))
  for(k in 2:n) {
    card <- cbind(card,0)
    card <- rbind(card, t(combn(n,k)))
  }
  
  # convert the cardinality table to binary equivalent and add conversion indices
  base.conv <- function(x,b) {
    out <- array(1,x)
    for(p in 2:x) out[p] <- b^{p-1}
    out
  }
  card.bits <- array(0,c(2^n,n))
  for(i in 1:(2^n)) for(j in 1:n) {
    if(card[i,j]>0) {card.bits[i,card[i,j]] <- 1}
  }
  card.bits <- cbind(card.bits,{1:{2^n}})
  card.bits <- cbind(card.bits,0)
  for(i in 1:(2^n)) {
    card.bits[i,(n+2)] <- 1+sum(base.conv(n,2)*card.bits[i,1:n])}
  
  
  # build constraints matrix 
  all.const <- array(0,0)
  # reordered g(x_i)
  for(k in 1:instances) {
    const.i <- array(0,numvars)
    for(s in 2:numvars) {
      const.i[s]<-min(the.data[k,card[s,]])
    }
    
    all.const <- rbind(all.const,const.i)
  }
  if(kadd>1){
    all.const <- cbind(all.const,-1*all.const[,(n+2):numvars])}
  # residual coefficients
  resid.pos <- -1*diag(instances)
  resid.neg <- diag(instances)
  # merge data constraints f - rij = y
  all.const <- cbind(all.const,resid.pos,resid.neg)
  # add row for mobius values sum to 1
  all.const<-rbind(all.const,c(array(1,numvars),array(-1,(numvars-n-1)),array(0,2*instances)))
  
  
  # add monotonicity constraints
  if(kadd>1) {
    num.monconst <-0
    for(m in (n+2):(2^n))	{
      
      setA <- subset(card[m,1:n],card[m,1:n]>0)
      # now find all subsets of corresponding set
      all.setB <- card.bits
      for(q in setdiff((1:n),setA)) {
        all.setB <- subset(all.setB,all.setB[,q]==0)}
      numv.setB <- subset(all.setB,all.setB[,(n+1)]<=numvars)
      for(b in setA) {
        mon.const.m <- array(0,ncol(all.const))
        mon.const.m[subset(all.setB[,(n+1)],all.setB[,b]>0)] <- 1
        mon.const.m[(numvars+1):(2*numvars-n-1)]<- -1*mon.const.m[(n+2):numvars]
        all.const<-rbind(all.const,mon.const.m)
        num.monconst<-num.monconst+1}
    }
  }
  
  all.const<-all.const[,2:(ncol(all.const))]
  
  # create rhs of constr
  constr.v <- array(0,nrow(all.const))
  
  for(i in 1:instances) {
    # populate with y observed
    constr.v[i] <- (the.data[i,ycol])
    # weights sum to 1
    constr.v[instances+1] <- 1					
    # remainder should stay 0
  }		
  
  # create inequalities direction vector   
  constr.d <- c(array("==",(instances+1)))
  
  if(kadd>1) {
    constr.d <- c(array("==",(instances+1)),array(">=",(num.monconst)))
  }
  
  # objective function is sum of resids
  obj.coeff <- c(array(0,(n)),array(1,2*instances))			
  if(kadd>1)	{
    obj.coeff <- c(array(0,(2*numvars-n-2)),array(1,2*instances))
  }
  # solve the lp to find w
  lp.output<-lp(direction="min",obj.coeff,all.const,constr.d,constr.v)$solution
  # create the weights matrix
  mob.weights<-array(lp.output[1:n])
  if(kadd>1) {									
    mob.weights<-c(array(lp.output[1:(n)]),lp.output[(n+1):(numvars-1)]-lp.output[(numvars):(2*numvars-n-2)])
  }
  zetaTrans <- function(x) {
    n <- log(length(x),2)
    zeta.out <- array(0,length(x))
    # first specify the correspond set
    for(i in 2:length(x))	{
      setA <- subset(card[i,1:n],card[i,1:n]>0)
      # now find all subsets of corresponding set
      all.setB <- cbind(card.bits,x)
      for(j in setdiff((1:n),setA)) {
        all.setB <- subset(all.setB,all.setB[,j]==0)}
      ZA <- 0
      # add each m(B) provided these have been attached in n+1 th position
      
      for(b in 1:nrow(all.setB))
        ZA <- ZA + all.setB[b,ncol(all.setB)]			
      zeta.out[i]<- ZA}
    zeta.out <- zeta.out[order(card.bits[,(n+2)])]
    zeta.out}
  
  mob.weights.v <- array(0,2^n)
  for(v in 2:length(mob.weights.v)) mob.weights.v[v]<-mob.weights[v-1]
  fm.weights.v <- zetaTrans(mob.weights.v)
  # calculate predicted values
  new.yvals <- array(0,instances)
  for(k in 1:instances) {
    new.yvals[k] <- choquet(as.numeric(the.data[k,1:n]),fm.weights.v[2:(2^n)])
  }
  # write the output
  
  write.table(cbind(the.data,new.yvals),output.1,row.names = FALSE, col.names = FALSE)
  # write some stats	
  RMSE <- (sum((new.yvals - the.data[,ycol])^2)/instances)^0.5
  av.l1error <- sum(abs(new.yvals - the.data[,ycol]))/instances
  shapley <- function(v) {    # 1. the input is a fuzzy measure
    n <- log(length(v)+1,2)     # 2. calculates n based on |v|
    shap <- array(0,n)          # 3. empty array for Shapley values
    for(i in 1:n) {             # 4. Shapley index calculation
      shap[i] <- v[2^(i-1)]*factorial(n-1)/factorial(n) # 
      # 4i.  empty set term
      for(s in 1:length(v)) {          # 4ii. all other terms
        if(as.numeric(intToBits(s))[i] == 0) {  #
          # 4iii.if i is not in set s
          S <- sum(as.numeric(intToBits(s)))    # 
          # 4iv. S is cardinality of s
          m <- (factorial(n-S-1)*factorial(S)/factorial(n))  # 
          # 4v. calculate multiplier 
          vSi <- v[s+2^(i-1)]          # 4vi. f-measure of s and i
          vS <- v[s]                   # 4vii. f-measure of s
          shap[i]<-shap[i]+m*(vSi-vS)  # 4viii. add term
        }                            #
      }                              #
    }                                #
    shap                               # 5. return shapley indices
  }                                  #    vector as output
  
  orness.v <- function(v) {     # 1. the input is a fuzzy measure
    n <- log(length(v)+1,2)       # 2. calculates n based on |v|
    m <- array(0,length(v))       # 3. empty array for multipliers
    for(i in 1:(length(v)-1)) {   # 4. S is the cardinality of 
      S <- sum(as.numeric(intToBits(i))) #    of the subset at v[i]
      m[i] <- factorial(n-S)*factorial(S)/factorial(n)  #
    }                             #
    sum(v*m)/(n-1)}                # 5. orness calculation
  
  
  somestats <- rbind(c("RMSE",RMSE),c("Av. abs error",av.l1error),c("Pearson Correlation",cor(new.yvals,the.data[,ycol])),c("Spearman Correlation",cor(new.yvals,the.data[,ycol],method="spearman")),c("Orness",orness.v(fm.weights.v[2:(2^n)])),c("i","Shapley i"),cbind(1:n,shapley(fm.weights.v[2:(2^n)])),c("binary number","fm.weights"),cbind(1:(2^n-1),fm.weights.v[2:(2^n)]))
  #card2 <- card
  #for(i in 1:nrow(card2)) for(j in 1:ncol(card2)) {if(card2[i,j]==0) card2[i,j]<- "" }
  #somestats <- cbind(somestats,rbind(array("",c(2,n)),c("sets",array("",(n-1))),card2[,1:n]))
  
  
  
  write.table(somestats,stats.1,quote = FALSE,row.names=FALSE,col.names=FALSE)	
}

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

# Print the row count before applying any filtering
print(paste("Dataset row count before filtering: ", nrow(the.data)))

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

# Print the number of rows after removal of outliers
print(paste("Dataset row count after filtering: ", nrow(filtered_data)))

num_samples <- nrow(filtered_data)

# Sample the data to have 450 rows
my.data <- filtered_data[sample(1:num_samples, num_row), c(1:num_col)]

# assert that we have no missing values
stopifnot(is.na(my.data) == FALSE)

# assert that we have no duplicates
stopifnot(nrow(my.data) == nrow(unique(my.data)))

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

# Check if variables are monotonically increasing
test_monoic <- function(data, x_id, y_id) {
  result <- monotonic(data, id.col=x_id, y.col=y_id, direction="inc")
  print(paste("Monotonicity increasing test for ", x_id))
  table(as.logical(result[,2]))
}

test_monoic(filtered_data, lr_temp_col_name, target_col_name)
test_monoic(filtered_data, lr_humidity_col_name, target_col_name)
test_monoic(filtered_data, or_temp_col_name, target_col_name)
test_monoic(filtered_data, or_humidity_col_name, target_col_name)
test_monoic(filtered_data, pressure_col_name, target_col_name)


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
generate_feature_stats <- function(data, # the data matrix
                                  col_names # array of column names
) {
  result_list <- list()
  for (col_name in col_names) {
    stats <- list(
      std = sd(data[, col_name]),
      mean = mean(data[, col_name]),
      min = min(data[, col_name]),
      max = max(data[, col_name])
    )
    result_list[[col_name]] = stats
  }
  result_list
}

# Calculate the zscore for a particular column
unit_zscore <- function(data,
                        col_name,
                        feature_stats) {
  0.15 * ((data - feature_stats[[col_name]][["mean"]]) / feature_stats[[col_name]][["std"]]) + 0.5
}


# Function to transform the matrix data
# arg: data - a data matrix with 5 columns (the last being the target)
# arg: feature_stats - descriptive statistics for features
# arg: apply_to_target - apply transform target or not
apply_distribution_transforms <- function(data,
                                          feature_stats,
                                          apply_to_target=TRUE) {
  #data[, lr_temp_col_name] <- 1 / (data[, lr_temp_col_name])
  data[, or_humidity_col_name] <- 1 / (data[, or_humidity_col_name] + 0.001) # constant added to avoid division by zero
  data[, pressure_col_name] <- (log(log(feature_stats[[pressure_col_name]][["max"]] + feature_stats[[pressure_col_name]][["min"]] - (data[, pressure_col_name]))))
  if (apply_to_target) {
    data[, target_col_name] <- log(data[, target_col_name])
  }
  data
  
}

# Apply unit zscore transformation to all columns / target
# arg: data - a matrix containing the dataset
# arg: feature_stats - a list of lists of the standard deviation and mean for 
#  the dataset for each  feature / target
# arg: apply_to_target - apply transform target or not
apply_unit_zscore_transforms <- function(data,
                                         feature_stats,
                                         apply_to_target=TRUE) {
  #data[, lr_temp_col_name] <- unit_zscore(data[, lr_temp_col_name], lr_temp_col_name, feature_stats)
  data[, lr_humidity_col_name] <- unit_zscore(data[, lr_humidity_col_name], lr_humidity_col_name, feature_stats)
  data[, or_temp_col_name] <- unit_zscore(data[, or_temp_col_name], or_temp_col_name, feature_stats)
  data[, or_humidity_col_name] <- unit_zscore(data[, or_humidity_col_name], or_humidity_col_name, feature_stats)
  data[, pressure_col_name] <- unit_zscore(data[, pressure_col_name], pressure_col_name, feature_stats)
  if (apply_to_target) {
    data[, target_col_name] <- unit_zscore(data[, target_col_name], target_col_name, feature_stats)
  }
  data
}

# Transform the data
raw_feature_stats <- generate_feature_stats(filtered_data, 
                                       c(
                                         #lr_temp_col_name,
                                         lr_humidity_col_name,
                                         or_temp_col_name,
                                         or_humidity_col_name,
                                         pressure_col_name,
                                         target_col_name))

transformed.data <- apply_distribution_transforms(my.data, raw_feature_stats)

transformed_feature_stats <- generate_feature_stats(transformed.data, 
                                            c(
                                              #lr_temp_col_name,
                                              lr_humidity_col_name,
                                              or_temp_col_name,
                                              or_humidity_col_name,
                                              pressure_col_name,
                                              target_col_name))

transformed.data <- apply_unit_zscore_transforms(transformed.data, transformed_feature_stats)

scaled_feature_stats <- generate_feature_stats(transformed.data, 
                                               c(
                                                 #lr_temp_col_name,
                                                 lr_humidity_col_name,
                                                 or_temp_col_name,
                                                 or_humidity_col_name,
                                                 pressure_col_name,
                                                 target_col_name))

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

# Check monotonically increasing after transformations
#test_monoic(transformed.data, lr_temp_col_name, target_col_name)
test_monoic(transformed.data, lr_humidity_col_name, target_col_name)
test_monoic(transformed.data, or_temp_col_name, target_col_name)
test_monoic(transformed.data, or_humidity_col_name, target_col_name)
test_monoic(transformed.data, pressure_col_name, target_col_name)

# Save the transformed data 
write.table(transformed.data, "name-transformed.txt")

# print raw descriptive stats
print("Raw descriptive stats")
print("---------------------")
print(raw_feature_stats)

# print transformed descriptive stats
print("Transformed descriptive stats")
print("-----------------------------")
print(transformed_feature_stats)

# print scaled descriptive stats
print("Scaled descriptive stats")
print("-----------------------------")
print(scaled_feature_stats)

################################################################################
# T3 - Model Construction
################################################################################

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
inference_matrix <- apply_distribution_transforms(inference_matrix, raw_feature_stats, apply_to_target=FALSE)
inference_matrix <- apply_unit_zscore_transforms(inference_matrix, transformed_feature_stats, apply_to_target=FALSE)

# Read the WAM weights from the file
wam_stats <- read.table("wam_statistics.txt", skip=5)

# Define the weights learnt via training of the Weighted Average Model
weights <- c(wam_stats$V2[wam_stats$V1 == 1],
             wam_stats$V2[wam_stats$V1 == 2],
             wam_stats$V2[wam_stats$V1 == 3],
             wam_stats$V2[wam_stats$V1 == 4])

print("Model Weights:")
print(weights)

# Use the Weighted Average Model to make the prediction
target_prediction_scaled <- QAM(inference_matrix, weights)

print(target_prediction_scaled)

# We have a target prediction but it is log transformed and scaled. We need to reverse this
target_prediciton = exp((transformed_feature_stats[[target_col_name]][["std"]] * (target_prediction_scaled - 0.5) / 0.15) 
                        + transformed_feature_stats[[target_col_name]][["mean"]])

print(target_prediciton)
