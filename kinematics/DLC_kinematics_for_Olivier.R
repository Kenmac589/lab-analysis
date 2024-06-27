# Where are your files located?
# Type the path here using this example as template
# If you are running this script from the same folder your data is in, use: data_path <- ""
data_path <- "Z:\\Olivier\\DLC_example\\"

# Function to calculate angles between vectors and/or versors
angle <- function(x,y) {
    temp   <- x%*%y # Dot product (scalar product)
    norm_x <- norm(x, ty="2")
    norm_y <- norm(y, ty="2")
    theta  <- acos(temp/(norm_x*norm_y))*180/pi
    as.numeric(theta)
}

# Define versors
ver_ho <- c(1, 0)           # Horizontal versor
ver_ve <- c(0, 1)           # Vertical versor

# Likelihood threshold to detect bad tracking
like_th <- 0.75

# Calibration parametres
# As of now, DeepLabCut does not give info about the video as an output
# Please take extra care to use the same resolution for the analysed videos
# and specify the number of pixels and frame rate in [frames/s] in the following lines
video_resolution_x <- 1280
video_resolution_y <- 600
frame_rate <- 500

# DeepLabCut calibration factors
# Insert calibration distances in [mm] considering the following scheme:
# 1  2
# 3  4
# 5  6
calibration_factors <- c(horiz=25,
                         vert=20,
                         diag=sqrt(20^2+25^2))

KINEMATICS <- list.files(data_path, pattern="DLC.*.csv")

mouse_data <- read.csv(paste0(data_path, "mouse_data.csv"))
mouse_data[] <- lapply(mouse_data, as.character)
mouse_data$Femur <- as.numeric(mouse_data$Femur)
mouse_data$Tibia <- as.numeric(mouse_data$Tibia)

cat("\n\nStart reading and calibrating files", sep="")

dir.create(file.path(data_path, "Analysis"), showWarnings=F)

for (trial in 1:length(KINEMATICS)) {
    
    cat("\nTrial ", trial, " out of ", length(KINEMATICS), sep="")
    
    # Preallocate joint angle vectors
    angle_hi <- numeric()      # Hip
    angle_kn <- numeric()      # Knee
    angle_an <- numeric()      # Ankle
    angle_to <- numeric()      # Toes
    angle_hi_sd <- numeric()
    angle_kn_sd <- numeric()
    angle_an_sd <- numeric()
    angle_to_sd <- numeric()
    
    # Read DLC file
    kin_data <- read.delim(paste0(data_path, KINEMATICS[trial]),
                           header=T, comment.char="", sep=",")
    
    temp_colnames <- paste(apply(kin_data[1, ], 1, as.character),
                           apply(kin_data[2, ], 1, as.character), sep="_")
    
    kin_data <- kin_data[-c(1, 2), ]
    colnames(kin_data) <- temp_colnames
    colnames(kin_data)[1] <- "frame"
    rownames(kin_data) <- NULL
    
    kin_data <- data.frame(apply(kin_data, 2, function(x) as.numeric(as.character(x))))
    
    # Lengths in [mm]
    mouse <- gsub("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_", "", KINEMATICS[trial])
    mouse <- gsub("_.*", "", mouse)
    femur_length <- mouse_data$Femur[grep(paste0("^", mouse, "$"), mouse_data$Code)]
    tibia_length <- mouse_data$Tibia[grep(paste0("^", mouse, "$"), mouse_data$Code)]
    
    # Try to fill gaps
    # Find marker names
    temp <- gsub("_x", "", colnames(kin_data))
    temp <- gsub("_y", "", temp)
    temp <- gsub("_likelihood", "", temp)
    temp <- temp[-grep("^frame$", temp)]
    markers <- unique(temp)
    
    # Fill each marker's trajectory gaps
    for (marker in markers) {
        data <- kin_data[, grep(paste0("^", marker), colnames(kin_data))]
        
        data_x <- data[, grep("_x", colnames(data))]
        data_y <- data[, grep("_y", colnames(data))]
        likely <- data[, grep("_likelihood", colnames(data))]
        
        plot_x <- data_x
        plot_y <- data_y
        
        # Search for likelihood lower than threshold like_th
        indexes <- which(likely<like_th)
        
        # Divide into intervals at least 20 frames long starting from when the likelihood drops
        # until it rises again and stays over like_th for at least 20 frames
        temp <- diff(indexes)
        temp[temp<20] <- 1
        temp <- which(temp>1)
        temp <- sort(c(temp, temp+1))
        temp <- c(1, temp, length(indexes))
        # Remove duplicates
        dup <- c(which(duplicated(temp)), which(duplicated(temp))-1)
        temp <- temp[-dup]
        # Remove intervals smaller than 20 frames
        indexes <- indexes[temp]
        small <- which(diff(indexes)<20)
        small <- c(small, small+1)
        indexes <- indexes[-small]
        
        if (length(indexes>0)) {
            for (index in seq(1, length(indexes), 2)) {
                start <- indexes[index]
                stop  <- indexes[index+1]
                
                xx <- data_x[start:stop]
                yy <- data_y[start:stop]
                
                outliers_xx <- boxplot.stats(xx, coef=0.0001)$out
                outliers_yy <- boxplot.stats(yy, coef=0.0001)$out
                
                # outliers_xx <- boxplot.stats(xx, coef=10000)$out
                # outliers_yy <- boxplot.stats(yy)$out
                #
                if (length(outliers_xx)>0) {
                    xx[unique(grep(paste0(outliers_xx, collapse="|"), xx))] <- NA
                }
                
                if (length(outliers_yy)>0) {
                    yy[unique(grep(paste0(outliers_yy, collapse="|"), yy))] <- NA
                }
                
                # par(mfrow=c(2, 1))
                
                temp_x <- spline(xx,
                                 n=length(data_x[start:stop]),
                                 method="fmm")
                temp_y <- spline(yy,
                                 n=length(data_y[start:stop]),
                                 method="fmm")
                
                data_x[start:stop] <- temp_x$y
                data_y[start:stop] <- temp_y$y
                
                # plot(data_x[start:stop], ty="l")
                # points(x=which(is.na(xx)),
                #        y=data_x[start:stop][which(is.na(xx))])
                # lines(temp_x$y, col=2)
                # plot(data_y[start:stop], ty="l")
                # points(x=which(is.na(yy)),
                #        y=data_y[start:stop][which(is.na(yy))])
                # lines(temp_y$y, col=2)
            }
            
            data[, grep("_x", colnames(data))] <- data_x
            data[, grep("_y", colnames(data))] <- data_y
            kin_data[, grep(paste0("^", marker), colnames(kin_data))] <- data
            
            # par(mfrow=c(2, 1))
            
            # plot(plot_x, ty="l", lwd=8, col="gray70",
            #      ylim=c(0, 1200),
            #      main=paste0(trial, " - ", marker))
            # lines(data_x, lwd=2)
            # plot(plot_y, ty="l", lwd=8, col="gray70",
            #      ylim=c(0, 850))
            # lines(data_y, lwd=2)
            # # plot(likely, ty="l", lwd=3)
        }
    }
    
    # Calibrate
    time <- kin_data[, "frame"]/frame_rate
    
    # Remove likelihood columns
    kin_data <- kin_data[, -grep("likelihood", colnames(kin_data))]
    
    # Change y-axis values to locate the origin in the bottom left corner of the video
    kin_data[, grep("_y$", colnames(kin_data))] <- abs(kin_data[, grep("_y$", colnames(kin_data))]-video_resolution_y)
    
    # Calibrate
    calib_set <- data.frame(kin_data[, grep("Calib", colnames(kin_data))])
    
    # Horizontal distances
    dist12 <- sqrt((calib_set$Calib1_x-calib_set$Calib2_x)^2+(calib_set$Calib1_y-calib_set$Calib2_y)^2)
    dist34 <- sqrt((calib_set$Calib3_x-calib_set$Calib4_x)^2+(calib_set$Calib3_y-calib_set$Calib4_y)^2)
    dist56 <- sqrt((calib_set$Calib5_x-calib_set$Calib6_x)^2+(calib_set$Calib5_y-calib_set$Calib6_y)^2)
    
    # Vertical distances
    dist13 <- sqrt((calib_set$Calib1_x-calib_set$Calib3_x)^2+(calib_set$Calib1_y-calib_set$Calib3_y)^2)
    dist35 <- sqrt((calib_set$Calib3_x-calib_set$Calib5_x)^2+(calib_set$Calib3_y-calib_set$Calib5_y)^2)
    dist24 <- sqrt((calib_set$Calib2_x-calib_set$Calib4_x)^2+(calib_set$Calib2_y-calib_set$Calib4_y)^2)
    dist46 <- sqrt((calib_set$Calib4_x-calib_set$Calib6_x)^2+(calib_set$Calib4_y-calib_set$Calib6_y)^2)
    
    # Diagonal distances
    dist14 <- sqrt((calib_set$Calib1_x-calib_set$Calib4_x)^2+(calib_set$Calib1_y-calib_set$Calib4_y)^2)
    dist23 <- sqrt((calib_set$Calib2_x-calib_set$Calib3_x)^2+(calib_set$Calib2_y-calib_set$Calib3_y)^2)
    dist36 <- sqrt((calib_set$Calib3_x-calib_set$Calib6_x)^2+(calib_set$Calib3_y-calib_set$Calib6_y)^2)
    dist45 <- sqrt((calib_set$Calib4_x-calib_set$Calib5_x)^2+(calib_set$Calib4_y-calib_set$Calib5_y)^2)
    
    distances <- data.frame(dist12, dist34, dist56,
                            dist13, dist35, dist24, dist46,
                            dist14, dist23, dist36, dist45)
    
    distances_av <- colMeans(distances)
    
    distances <- c(horiz=mean(distances_av[1:3]),
                   vert=mean(distances_av[4:7]),
                   diag=mean(distances_av[8:11]))
    
    calibration_matrix <- data.frame(mm=calibration_factors,
                                     pixels=distances,
                                     mm.pixel=calibration_factors/distances)
    
    calibration_factor <- mean(calibration_matrix$mm.pixel)
    
    # Remove calibration columns and calibrate the rest
    kin_data <- kin_data[, -grep("Calib", colnames(kin_data))]
    kin_data <- kin_data*calibration_factor
    
    kin_data[, "frame"] <- time
    colnames(kin_data)[colnames(kin_data)=="frame"] <- "time"
    
    for (ss in 1:nrow(kin_data)) {
        # 2D coordinates of landmarks
        coord_il <- c(kin_data$IliacCrest_x[ss], kin_data$IliacCrest_y[ss]) # Iliac crest
        coord_hi <- c(kin_data$Hip_x[ss],        kin_data$Hip_y[ss])        # Hip
        coord_an <- c(kin_data$Ankle_x[ss],      kin_data$Ankle_y[ss])      # Ankle
        coord_mt <- c(kin_data$Metatarsal_x[ss], kin_data$Metatarsal_y[ss]) # Metatarsophalangeal (MTP) joint
        coord_to <- c(kin_data$ToeTip_x[ss],     kin_data$ToeTip_y[ss])     # Toe tip
        
        # Calculate virtual knee coordinates
        # Consider two circles with centers:
        # (kin_data$Hip_x[ss], kin_data$Hip_y[ss]) and (kin_data$Ankle_x[ss], kin_data$Ankle_y[ss])
        # The radii are the length of the femur and of the tibia, respectively
        # The distance between hip and ankle is
        dist_hi_an <- norm(coord_hi-coord_an, ty="2")
        
        if (dist_hi_an>(femur_length+tibia_length) ||
            dist_hi_an==0 ||
            dist_hi_an<abs(femur_length-tibia_length)) {
            # cat("\nFemur and tibia lengths are too short!!!")
            
            angle_hi[ss] <- NA
            angle_kn[ss] <- NA
            angle_an[ss] <- NA
            angle_to[ss] <- NA
            
            next
        } else {
            # Coordinates of the point along the line between ankle and hip
            # Look here for details: http://paulbourke.net/geometry/circlesphere/
            aa <- (femur_length^2-tibia_length^2+dist_hi_an^2)/(2*dist_hi_an)
            temp_point <- coord_hi+aa*(coord_an-coord_hi)/dist_hi_an
            hh <- sqrt(femur_length^2-aa^2)
            
            # There are two possible solutions, but we will take only the good one
            # (where the knee does not go over maximal extension, so the one with the smallest x)
            decision <- numeric()
            
            decision[1] <- temp_point[1]+hh*(coord_hi[2]-coord_an[2])/dist_hi_an
            decision[2] <- temp_point[1]-hh*(coord_hi[2]-coord_an[2])/dist_hi_an
            
            coord_kn <- numeric()
            if (decision[1]>decision[2]) {
                # Solution 1
                coord_kn[1] <- temp_point[1]+hh*(coord_hi[2]-coord_an[2])/dist_hi_an
                coord_kn[2] <- temp_point[2]-hh*(coord_hi[1]-coord_an[1])/dist_hi_an
            } else if (decision[1]<decision[2]) {
                # Solution 1
                coord_kn[1] <- temp_point[1]-hh*(coord_hi[2]-coord_an[2])/dist_hi_an
                coord_kn[2] <- temp_point[2]+hh*(coord_hi[1]-coord_an[1])/dist_hi_an
            }
        }
        
        # Hip angle (angle between pelvis and femur, 90°=standing position, 0°=hip flexed)
        # Compute vectors
        vec_pe <- coord_il-coord_hi   # This is the pelvis
        vec_fe <- coord_kn-coord_hi   # This is the femur
        if (any(is.na(vec_pe))) {
            angle_hi[ss] <- NA
        } else if (any(is.na(vec_fe))) {
            angle_hi[ss] <- NA
        }  else {
            # Compute angle between the two vectors
            angle_hi[ss] <- angle(vec_pe, vec_fe)
        }
        
        # Knee angle (angle between femur and tibia, 0°=knee extended, 90°=knee flexed)
        # Compute vectors
        vec_fe <- coord_hi-coord_kn   # This is the femur
        vec_ti <- coord_an-coord_kn   # This is the tibia
        if (any(is.na(vec_fe))) {
            angle_kn[ss] <- NA
        } else if (any(is.na(vec_ti))) {
            angle_kn[ss] <- NA
        } else {
            # Compute angle between the two vectors
            angle_kn[ss] <- angle(vec_fe, vec_ti)
        }
        
        # Ankle angle (angle between tibia and paw until MTP, <90°=paw dorsiflexed, >90°=paw plantarflexed)
        # Compute vectors
        vec_ti <- coord_kn-coord_an   # This is the tibia
        vec_am <- coord_mt-coord_an   # This is the paw from ankle to metatarsal
        if (any(is.na(vec_ti))) {
            angle_an[ss] <- NA
        } else if (any(is.na(vec_am))) {
            angle_an[ss] <- NA
        } else {
            # Compute angle between the two vectors
            angle_an[ss] <- angle(vec_ti, vec_am)
        }
        
        # MTP angle (angle of the tows, <90°=toes flexed, >90°=toes extended)
        # Compute vectors
        vec_am <- coord_an-coord_mt   # This is the paw from ankle to metatarsal
        vec_mt <- coord_to-coord_mt   # This is the paw from MTP (metatarsophalangeal joint) to toe tip
        if (any(is.na(vec_am))) {
            angle_to[ss] <- NA
        } else if (any(is.na(vec_mt))) {
            angle_to[ss] <- NA
        } else {
            # Compute angle between the two vectors
            angle_to[ss] <- angle(vec_am, vec_mt)
        }
    }
    
    # Displacements and angles matrix in [mm] and [°]
    kin_data <- data.frame(kin_data,
                           Hip_angle=angle_hi,
                           Knee_angle=angle_kn,
                           Ankle_angle=angle_an,
                           MTP_angle=angle_to)
    
    write.csv(kin_data,
              file=paste0(data_path, "Analysis\\",
                          gsub("DLC.*", "", KINEMATICS[trial]), "_calibrated.csv"),
              quote=F, row.names=F)
}