# Install (if needed) required packages ----

pkgs_list <- c("parallel",
               # "plyr",
               # "ggplot2",
               # "gridExtra",
               # "Cairo",
               "signal")
pkgs_new <- pkgs_list[!(pkgs_list %in% installed.packages()[,"Package"])]
if(length(pkgs_new)) install.packages(pkgs_new)

# Load required packages
lapply(pkgs_list, library, character.only=T)
rm(list=setdiff(ls(), c("cl")))

# Create cluster for parallel computing if not already done ----
clusters <- objects()

if (sum(grepl("cl", clusters))==0) {
    # Decide how many processor threads have to be excluded from the cluster
    # It is a good idea to leave at least one free, so that the machine can be used during computation
    free_cores <- 1
    cl <- makeCluster(detectCores()-free_cores)
}

# Load data ----
load("CYCLE_TIMES.RData")
load("RAW_EMG.RData")

# Global variables
HPo    <- 4             # High-pass filter order
HPf    <- 50            # High-pass filter frequency [Hz]
LPo    <- HPo           # Low-pass filter order
LPf    <- 30            # Low-pass filter frequency [Hz]
points <- 200           # Walking or swimming cycle length (interpolated points)
cycles <- numeric()     # To save number of cycles considered

# Preallocate for writing results
FILT_EMG   <- vector("list", length(RAW_EMG))
list_names <- character(length=length(RAW_EMG))

for (ii in 1:length(RAW_EMG)) {
    trial     <- names(RAW_EMG[ii])
    condition <- substr(trial, 15, 15)
    emg_data  <- RAW_EMG[[ii]]
    emg_data_temp <- emg_data
    
    # Filtering
    # EMG system acquisition frequency [Hz]
    freq <- round(nrow(emg_data)/(tail(emg_data$Time, 1)+emg_data$Time[2]), 0)
    
    # High-pass IIR (Infinite Impulse Response) Butterworth zero-phase filter design
    # Critical frequencies must be between 0 and 1, where 1 is the Nyquist frequency
    # "filtfilt" is for zero-phase filtering
    HPfn <- HPf/(freq/2)            # Normalise by the Nyquist frequency (f/2)
    HP   <- butter(HPo, HPfn, type="high")
    for (kk in 2:ncol(emg_data_temp)) {
        emg_data_temp[,kk] <- filtfilt(HP, emg_data_temp[,kk])
    }
    
    # Full-wave rectification
    emg_data_temp <- abs(emg_data_temp)
    
    # Low-pass IIR (Infinite Impulse Response) Butterworth zero-phase filter design
    # Critical frequencies must be between 0 and 1, where 1 is the Nyquist frequency
    # "filtfilt" is for zero-phase filtering
    LPfn <- LPf/(freq/2)            # Normalise by the Nyquist frequency (f/2)
    LP   <- butter(LPo, LPfn, type="low")
    for (kk in 2:ncol(emg_data_temp)) {
        emg_data_temp[,kk] <- filtfilt(LP, emg_data_temp[,kk])
    }
    
    emg_data_temp[emg_data_temp<0] <- 0                 # Set negative values to zero
    temp <- emg_data_temp
    temp[temp==0] <- Inf
    emg_data_temp[emg_data_temp==0] <- min(temp)        # Set the zeros to the smallest nonzero entry in V
    emg_data_temp$Time <- emg_data$Time
    
    emg_data_filt <- emg_data_temp[,-1]
    
    # Isolate cycles and normalise time to (points) points
    # In wallking, this is (points/2 stance, points/2 swing)
    # In swimming, there is only one cycle section
    c_time <- CYCLE_TIMES[[grep(gsub("RAW_EMG_", "", trial), names(CYCLE_TIMES))]]
    
    cycs <- nrow(c_time)
    cycs_list <- list()               # To store the isolated cycles
    
    # Isolate each cycle
    cyc <- 0
    
    emg_data_filt <- data.frame(emg_data$Time, emg_data_filt)
    colnames(emg_data_filt)[1] <- "Time"
    
    muscles <- colnames(emg_data_temp)
    
    if (condition=="S") {
        for (jj in 1:(cycs-1)) {
            # Swim cycle
            temp <- data.frame()
            t1   <- round(c_time$onset[jj]*freq+1, 0)
            t2   <- round(c_time$onset[jj+1]*freq+1, 0)
            
            if (t1>nrow(emg_data_filt) || t2>nrow(emg_data_filt)) {
                cycs <- jj-1
                break
            } else {
                temp  <- emg_data_filt[t1:t2, -1]
            }
            
            # Check if there is data
            if (sum(temp, na.rm=T)==0) next
            
            temp1 <- numeric()
            temp2 <- data.frame(matrix(1:points, nrow=points, ncol=1))
            
            # Interpolate each channel to (points) points
            for (kk in 1:ncol(temp)) {
                temp1 <- as.data.frame(approx(temp[,kk], method="linear", n=points))
                temp1 <- temp1[,2]
                temp2 <- cbind(temp2, temp1)
            }
            colnames(temp2) <- muscles
            
            # Set every value >1 to 1
            temp2[temp2>1] <- 1
            temp2$Time     <- c(1:points)
            
            # For concatenating the data
            if (jj==1) {
                emg_data_co <- temp2
            } else {
                emg_data_co <- rbind(temp2, emg_data_co)
            }
            cyc <- cyc+1
        }
    } else if (condition=="W") {
        for (jj in 1:(cycs-1)) {
            # Stance
            temp <- data.frame()
            t1   <- which(emg_data_filt$Time>c_time$td[jj])[1]
            t2   <- which(emg_data_filt$Time>c_time$to[jj])[1]
            
            if (t1>nrow(emg_data_filt) || t2>nrow(emg_data_filt)) {
                cycs <- jj-1
                break
            } else {
                temp <- emg_data_filt[t1:t2, -1]
            }
            
            temp1 <- numeric()
            temp2 <- data.frame(matrix(1:points, nrow=points/2, ncol=1))
            
            # Interpolate each channel to (points/2) points
            for (kk in 1:ncol(temp)) {
                temp1 <- as.data.frame(approx(temp[,kk], method="linear", n=points/2))
                temp1 <- temp1[,2]
                temp2 <- cbind(temp2, temp1)
            }
            colnames(temp2) <- muscles
            
            # Swing
            temp <- data.frame()
            # t3   <- round(c_time$td[jj]*freq+1, 0)
            t3   <- t2+1
            t4   <- which(emg_data_filt$Time>c_time$td[jj+1])[1]-1
            temp <- emg_data_filt[t3:t4, -1]
            # Check if there is data
            if (sum(temp, na.rm=T)==0) next
            temp1 <- numeric()
            temp3 <- data.frame(matrix(1:points, nrow=points/2, ncol=1))
            # Interpolate each channel to (points/2) points
            for (kk in 1:ncol(temp)) {
                temp1 <- as.data.frame(approx(temp[,kk], method="linear", n=points/2))
                temp1 <- temp1[,2]
                temp3 <- cbind(temp3, temp1)
            }
            colnames(temp3) <- muscles
            
            temp4 <- rbind(temp2, temp3)
            
            # Set every value >1 to 1
            temp4[temp4>1] <- 1
            temp4$Time     <- c(1:points)
            
            # For concatenating the data
            if (jj==1) {
                emg_data_co <- temp4
            } else {
                emg_data_co <- rbind(emg_data_co, temp4)
            }
            cyc <- cyc+1
        }
    }
    
    time <- emg_data_co$Time
    
    # Minimum subtraction
    emg_data_co <- data.frame(apply(emg_data_co, 2, function(x) x-min(x)))
    # Amplitude normalisation
    emg_data_co <- data.frame(apply(emg_data_co, 2, function(x) x/max(x)))
    
    emg_data_co$Time <- time
    
    FILT_EMG[[ii]] <- emg_data_co
    names(FILT_EMG[[ii]]) <- muscles
    list_names[ii] <- gsub("RAW_EMG", "FILT_EMG", trial)
    
    cat("\n        Progress: ", round(100*(ii)/(length(RAW_EMG)), d=1), "% completed",
        "\n           Trial: ", trial,
        "\n      Trial num.: ", ii, " of ", length(RAW_EMG),
        "\n  Acq. frequency: ", freq,
        "\n HP and LP order: ", HPo,
        "\n        HP freq.: ", HPf,
        "\n        LP freq.: ", LPf,
        "\n          Cycles: ", cycs, sep="", "\n")
}

names(FILT_EMG) <- list_names
save(FILT_EMG, file="FILT_EMG.RData")

# Export to ASCII
for (ii in 1:length(FILT_EMG)) {
    write.table(FILT_EMG[[ii]], file=paste0(names(FILT_EMG[ii]), ".dat"), sep="\t")
    cat("\nExporting filtered EMG file ", ii, " of ", length(FILT_EMG), sep="")
}

# Synergies extraction ----
ll <- length(FILT_EMG)
source("fun_synsNMFn.R")

cat("\n\nExtract synergies", sep="", "\n")

tictoc <- system.time({
    # "synsNMFn" is the core function for extracting synergies
    SYNS <- parLapply(cl, FILT_EMG, synsNMFn)
    
    # - - - - - - - - - - - - - - - - - - - - - - -
    #     
    #     Windows >= 8 x64 build 9200 
    # R version 3.5.1 (2018-07-02)
    # 4 cores, 8 logical
    # 
    # Number of trials: 2
    # Computation time (GNMF, filt): 7.86 s
    # Average trial comp. time: 3.93 s
    # 
    # - - - - - - - - - - - - - - - - - - - - - - -
    
})

names(SYNS) <- gsub("FILT_EMG", "SYNS", names(SYNS))

cat("- - - - - - - - - - - - - - - - - - - - - - -\n",
    "\n", Sys.info()[[1]], " ", Sys.info()[[2]], " ", Sys.info()[[3]], " ",
    "\n", R.version$version.string,
    "\n", detectCores(logical=F), " cores, ", detectCores(logical=T), " logical",
    "\n\n               Number of trials: ", ll,
    "\n  Computation time (GNMF, filt): ", round(tictoc[[3]], d=2), " s",
    "\n       Average trial comp. time: ", round(tictoc[[3]]/ll, d=2), " s\n",
    "\n- - - - - - - - - - - - - - - - - - - - - - -", sep="")

save(SYNS, file="SYNS.RData")

# Results preallocations
SYNS_names <- names(SYNS)
SYNS_W     <- list()
SYNS_H     <- list()

# Prepare results, reading from lists returned by SYNS functions
for (ii in 1:length(SYNS_names)) {
    temp         <- SYNS[[ii]]
    SYNS_W[[ii]] <- temp$W
    SYNS_H[[ii]] <- temp$H
    rownames(SYNS_H[[ii]]) <- paste0("Syn", c(1:nrow(SYNS_H[[ii]])))
    colnames(SYNS_W[[ii]]) <- paste0("Syn", c(1:ncol(SYNS_W[[ii]])))
}
names(SYNS_H) <- gsub("SYNS", "SYNS_H", names(SYNS))
names(SYNS_W) <- gsub("SYNS", "SYNS_W", names(SYNS))

save(SYNS_H, file="SYNS_H.RData")
save(SYNS_W, file="SYNS_W.RData")

# Export to ASCII
for (ii in 1:length(SYNS_H)) {
    write.table(SYNS_H[[ii]], file=paste0(names(SYNS_H[ii]), ".dat"), sep="\t")
    cat("\nExporting primitives file ", ii, " of ", length(SYNS_H), sep="")
}

# Export to ASCII
for (ii in 1:length(SYNS_W)) {
    write.table(SYNS_W[[ii]], file=paste0(names(SYNS_W[ii]), ".dat"), sep="\t")
    cat("\nExporting modules file ", ii, " of ", length(SYNS_W), sep="")
}
