library(perfectphyloR)
 # Haplotype matrix
 haplo_mat <- matrix(c(1,1,1,0,
                       0,0,0,0,
                       1,1,1,1,
                       1,0,0,0), byrow = TRUE, ncol = 4)
 # SNV names
 SNV_names <- c(paste("SNV", 1:4, sep = ""))
 # Haplotype names
 hap_names <- c("h1", "h2", "h3", "h4")
 # SNV positions in base pairs
 SNV_posns <- c(1000, 2000, 3000, 4000)
 ex_hapMat <- createHapMat(hapmat = haplo_mat,
                           snvNames = SNV_names,
                           hapNames = hap_names,
                           posns = SNV_posns)

rdend <- reconstructPP(hapMat = ex_hapMat,
                        focalSNV = 2,
                        minWindow = 1,
                        sep = "-")