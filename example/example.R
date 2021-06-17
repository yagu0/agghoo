library(agghoo)

data(iris) #already there
library(mlbench)
data(PimaIndiansDiabetes)

# Run only agghoo on iris dataset (split into train/test, etc).
# Default parameters: see ?agghoo and ?AgghooCV
compareTo(iris[,-5], iris[,5], agghoo_run)

# Run both agghoo and standard CV, specifiying some parameters.
compareTo(iris[,-5], iris[,5], list(agghoo_run, standardCV_run), gmodel="tree")
compareTo(iris[,-5], iris[,5], list(agghoo_run, standardCV_run),
          gmodel="knn", params=c(3, 7, 13, 17, 23, 31),
          CV = list(type="vfold", V=5, shuffle=T))

# Run both agghoo and standard CV, averaging errors over N=10 runs
# (possible for a single method but wouldn't make much sense...).
compareMulti(PimaIndiansDiabetes[,-9], PimaIndiansDiabetes[,9],
             list(agghoo_run, standardCV_run), N=10, gmodel="rf")

# Compare several values of V
compareRange(PimaIndiansDiabetes[,-9], PimaIndiansDiabetes[,9],
             list(agghoo_run, standardCV_run), N=10, V_range=c(10, 20, 30))

# For example to use average of squared differences.
# Default is "mean(abs(y1 - y2))".
loss2 <- function(y1, y2) mean((y1 - y2)^2)

# In regression on artificial datasets (TODO: real data?)
data <- mlbench.twonorm(300, 3)$x
target <- rowSums(data)
compareMulti(data, target, list(agghoo_run, standardCV_run),
             N=10, gmodel="tree", params=c(1, 3, 5, 7, 9), loss=loss2,
             CV = list(type="MC", V=12, test_size=0.3))

compareMulti(data, target, list(agghoo_run, standardCV_run),
             N=10, floss=loss2, CV = list(type="vfold", V=10, shuffle=F))

# Random tests to check that method doesn't fail in 1D case
M <- matrix(rnorm(200), ncol=2)
compareTo(as.matrix(M[,-2]), M[,2], list(agghoo_run, standardCV_run), gmodel="knn")
compareTo(as.matrix(M[,-2]), M[,2], list(agghoo_run, standardCV_run), gmodel="tree")
