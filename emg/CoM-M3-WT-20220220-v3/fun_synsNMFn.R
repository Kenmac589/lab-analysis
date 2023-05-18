synsNMFn <- function (V, ...)
{
    r <- 1                                      # Initialise number of synergies    
    R2_target <- 0.01                           # Convergence criterion (percent of the R2 value)
    R2_cross <- numeric()                       # R2 values for cross validation and syns number asessment
    AIC_cross <- numeric()                      # AIC values for cross validation and syns number assessment
    W_list <- list()                            # To save factorisation W matrices (synergies)
    H_list <- list()                            # To save factorisation H matrices (primitives)
    Vr_list <- list()                           # To save factorisation Vr matrices (reconstructed signals)
    iterations <- numeric()                     # To save the iterations number
    results <- list()                           # To save the final results
    
    # Original matrix
    V <- as.matrix(V)
    V <- t(V[,-1])                              # Needs to be transposed for NMF
    V[V<0] <- 0                                 # Set negative values to zero
    temp <- V
    temp[temp==0] <- Inf
    V[V==0] <- min(temp, na.rm=T)               # Set the zeros to the smallest nonzero entry in V
    
    m <- nrow(V)                                # Number of muscles
    n <- ncol(V)                                # Number of time points
    
    max_syns <- m-round(m/4, 0)              # Max number of syns is m-(m/4)
    
    for (r in r:max_syns) {                     # Run NMF with different initial conditions (syns num.)
        psi <- (m+n)*r                          # Total number of estimated parametres
        R2_choice  <- numeric()                 # Collect the R2 values for each syn and choose the max
        AIC_choice <- numeric()                 # Collect the AIC values for each syn
        
        for (j in 1:10) {                       # Run NMF 10 times for each syn and choose best run
            # To save error values
            R2  <- numeric()                    # 1st cost function (R squared)
            AIC <- numeric()                    # 2nd cost function (Akaike Info. Criterion)
            SST <- numeric()                    # Total sum of squares
            RSS <- numeric()                    # Residual sum of squares or min. reconstr. error
            
            # Initialise iterations and define max number of iterations
            iter <- 1
            max_iter <- 1000
            # Initialise the two factorisation matrices with random values (uniform distribution)
            H <- matrix(runif(r*n, min=0.01, max=1), nrow=r, ncol=n)
            W <- matrix(runif(m*r, min=0.01, max=1), nrow=m, ncol=r)

            # Iteration zero
            H <- H * (t(W) %*% V) / (t(W) %*% W %*% H)
            W <- W * (V %*% t(H)) / (W %*% H %*% t(H))
            Vr <- W %*% H                       # Reconstructed matrix
            RSS <- sum((V-Vr)^2)
            SST <- sum((V-mean(V))^2)
            R2[iter] <- 1-(RSS/SST)
            
            # l2-norm normalisation which eliminates trivial scale indeterminacies
            # The cost function doesn't change. Impose ||W||2 = 1 and normalise H accordingly.
            # ||W||2, also called L2,1 norm or l2-norm, is a sum of Euclidean norm of columns.
            for (kk in 1:r) {
                norm <- sqrt(sum(W[,kk]^2))
                W[,kk] <- W[,kk]/norm
                H[kk,] <- H[kk,]*norm
            }
            # Slower, but equivalent, approach (on a 32760x254 matrix 0.12 s vs. 0.37 s):
            # for (kk in 1:r) {
            #         norm <- sum(norm(W[,kk], "2"))
            #         W[,kk] <- W[,kk]/norm
            #         H[kk,] <- H[kk,]*norm
            #     }
            # Slower, but equivalent, approach (on a 32760x254 matrix 0.12 s vs. 0.47 s):
            # norm <- apply(W, 2, function(x) sqrt(sum(x^2)))
            #     for (kk in 1:length(norm)) {
            #         W[,kk] <- W[,kk]/norm[kk]
            #         H[kk,] <- H[kk,]*norm
            #     }
            
            # Start iterations for NMF convergence
            for (iter in iter:max_iter)  {
                H <- H * (t(W) %*% V) / (t(W) %*% W %*% H)
                W <- W * (V %*% t(H)) / (W %*% H %*% t(H))
                Vr <- W %*% H
                RSS <- sum((V-Vr)^2)
                SST <- sum((V-mean(V))^2)
                R2[iter] <- 1-(RSS/SST)
                
                # l2-norm normalisation
                for (kk in 1:r) {
                    norm <- sqrt(sum(W[,kk]^2))
                    W[,kk] <- W[,kk]/norm
                    H[kk,] <- H[kk,]*norm
                }
                
                # Check if the increase of R2 in the last 20 iterations is less than the target
                if (iter>20) {
                    R2_diff <- R2[iter]-R2[iter-20]
                    if (R2_diff<R2[iter]*R2_target/100) {
                        break
                    }
                }
            }
            R2_choice[j] <- R2[iter]
            AIC_choice[j] <- 2*(RSS+psi)
        }
        fit <- cbind(R2_choice, AIC_choice)
        R2_cross[r] <- max(fit[,1])
        nr_cross <- which(fit==max(fit[,1]), arr.ind = TRUE)
        AIC_cross[r] <- fit[nr_cross[1,1],2]
        W_list[[r]] <- as.data.frame(W)
        H_list[[r]] <- as.data.frame(H)
        Vr_list[[r]] <- as.data.frame(Vr)
        iterations[r] <- iter
    }
    
    # Choose the minimum number of synergies using the AIC criterion
    syns_data <- cbind(c(1:r), AIC_cross)
    nr_syns <- which(syns_data==min(syns_data[,2]), arr.ind = TRUE)
    syns_AIC <- as.numeric(syns_data[nr_syns[1,1],1])
    
    # Choose the minimum number of synergies using the R2 criterion
    MSE <- 100                                  # Initialise the Mean Squared Error (MSE)
    iter <- 0                                   # Initialise iterations
    while (MSE>1e-04) {
        iter <- iter+1
        if (iter==r-1) {
            break
        }
        R2_interp <- as.data.frame(cbind(c(1:(r-iter+1)), R2_cross[iter:r]))
        colnames(R2_interp) <- c("synergies", "R2_values")
       n <- nrow(R2_interp)
        linear <- lm(R2_values ~ synergies, R2_interp)
        linear_points <- linear[[5]]
        MSE <- sum((linear_points-R2_interp[,2])^2)/n
    }
    syns_R2 <- iter
    
    results[[1]] <- as.numeric(syns_R2)
    results[[2]] <- as.numeric(syns_AIC)
    results[[3]] <- W_list[[syns_R2]]
    results[[4]] <- H_list[[syns_R2]]
    results[[5]] <- Vr_list[[syns_R2]]
    results[[6]] <- as.numeric(iterations[syns_R2])
    results[[7]] <- as.numeric(R2_cross[syns_R2])
    names(results) <- c("synsR2", "synsAIC", "W", "H", "Vr", "iterations", "R2")
    
    return(results)
}
