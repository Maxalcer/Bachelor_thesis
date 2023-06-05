library(perfectphyloR)
library(lambda.r)

lines <- readLines("train_inf.txt")

tables <- list()
mat <- c()

for (i in 1:length(lines)){
  
  if(i %% 11 != 0){
    line <- as.integer(unlist(strsplit(lines[i], "")))
    mat <- c(mat, line)
  }
  else{
    mat <- matrix(mat, byrow = TRUE, ncol = 10)
    tables <- append(tables, list(mat))
    mat <- c()
  }
  
}
mat <- unique(matrix(mat, byrow = TRUE, ncol = 10))
tables <- append(tables, list(mat))

SNV_names <- c(paste("SNV", 1:10, sep = ""))
SNV_posns <- c(1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 100000)

count <- 0

for (tab in tables){
  hap_names <- c(paste("h", 1:nrow(tab), sep = ""))
  ex_hapMat <- createHapMat(hapmat = tab,
                            snvNames = SNV_names,
                            hapNames = hap_names,
                            posns = SNV_posns)

  rdend <- reconstructPP(hapMat = ex_hapMat,
                         focalSNV = 10,
                         minWindow = 1,
                         sep = "-")

is_fin <- FALSE

temp <- grepl("-", rdend[[3]], fixed = TRUE)

for (t in temp){
  is_fin <- is_fin || t
}

if(is_fin){
  count <- count + 1
  }
}

print(paste("number of Matrices with Infinite Sites violation: ", count))



