standardCV_core <- function(data, target, task = NULL, gmodel = NULL, params = NULL,
  loss = NULL, CV = list(type = "MC", V = 10, test_size = 0.2, shuffle = TRUE)
) {
  if (!is.null(task))
    task = match.arg(task, c("classification", "regression"))
  if (is.character(gmodel))
    gmodel <- match.arg(gmodel, c("knn", "ppr", "rf", "tree"))
  if (is.numeric(params) || is.character(params))
    params <- as.list(params)
  if (is.null(task)) {
    if (is.numeric(target))
      task = "regression"
    else
      task = "classification"
  }

  if (is.null(loss)) {
    loss <- function(y1, y2) {
      if (task == "classification") {
        if (is.null(dim(y1)))
          mean(y1 != y2)
        else {
          if (!is.null(dim(y2)))
            mean(rowSums(abs(y1 - y2)))
          else {
            y2 <- as.character(y2)
            names <- colnames(y1)
            positions <- list()
            for (idx in seq_along(names))
              positions[[ names[idx] ]] <- idx
            mean(vapply(
              seq_along(y2),
              function(idx) sum(abs(y1[idx,] - positions[[ y2[idx] ]])),
              0))
          }
        }
      }
      else
        mean(abs(y1 - y2))
    }
  }

  n <- nrow(data)
  shuffle_inds <- NULL
  if (CV$type == "vfold" && CV$shuffle)
    shuffle_inds <- sample(n, n)
  get_testIndices <- function(v, shuffle_inds) {
    if (CV$type == "vfold") {
      first_index = round((v-1) * n / CV$V) + 1
      last_index = round(v * n / CV$V)
      test_indices = first_index:last_index
      if (!is.null(shuffle_inds))
        test_indices <- shuffle_inds[test_indices]
    }
    else
      test_indices = sample(n, round(n * CV$test_size))
    test_indices
  }
  list_testinds <- list()
  for (v in seq_len(CV$V))
    list_testinds[[v]] <- get_testIndices(v, shuffle_inds)

  gmodel <- agghoo::Model$new(data, target, task, gmodel, params)
  best_error <- Inf
  best_model <- NULL
  for (p in seq_len(gmodel$nmodels)) {
    error <- 0
    for (v in seq_len(CV$V)) {
      testIdx <- list_testinds[[v]]
      dataHO <- data[-testIdx,]
      testX <- data[testIdx,]
      targetHO <- target[-testIdx]
      testY <- target[testIdx]
      if (!is.matrix(dataHO) && !is.data.frame(dataHO))
        dataHO <- as.matrix(dataHO)
      if (!is.matrix(testX) && !is.data.frame(testX))
        testX <- as.matrix(testX)
      model_pred <- gmodel$get(dataHO, targetHO, p)
      prediction <- model_pred(testX)
      error <- error + loss(prediction, testY)
    }
    if (error <= best_error) {
      newModel <- list(model=model_pred, param=gmodel$getParam(p))
      if (error == best_error)
        best_model[[length(best_model)+1]] <- newModel
      else {
        best_model <- list(newModel)
        best_error <- error
      }
    }
  }
  best_model[[ sample(length(best_model), 1) ]]
}

standardCV_run <- function(
  dataTrain, dataTest, targetTrain, targetTest, verbose, CV, floss, ...
) {
  s <- standardCV_core(dataTrain, targetTrain, ...)
  if (verbose)
    print(paste( "Parameter:", s$param ))
  ps <- s$model(test)
  err_s <- floss(ps, targetTest)
  if (verbose)
    print(paste("error CV:", err_s))
  invisible(c(errors, err_s))
}

agghoo_run <- function(
  dataTrain, dataTest, targetTrain, targetTest, verbose, CV, floss, ...
) {
  a <- agghoo(dataTrain, targetTrain, ...)
  a$fit(CV)
  if (verbose) {
    print("Parameters:")
    print(unlist(a$getParams()))
  }
  pa <- a$predict(dataTest)
  err <- floss(pa, targetTest)
  if (verbose)
    print(paste("error agghoo:", err))
}

# ... arguments passed to agghoo or any other procedure
compareTo <- function(
  data, target, rseed=-1, verbose=TRUE, floss=NULL,
  CV = list(type = "MC",
            V = 10,
            test_size = 0.2,
            shuffle = TRUE),
  method_s=NULL, ...
) {
  if (rseed >= 0)
    set.seed(rseed)
  n <- nrow(data)
  test_indices <- sample( n, round(n / ifelse(n >= 500, 10, 5)) )
  trainData <- as.matrix(data[-test_indices,])
  trainTarget <- target[-test_indices]
  testData <- as.matrix(data[test_indices,])
  testTarget <- target[test_indices]

  # Set error function to be used on model outputs (not in core method)
  if (is.null(floss)) {
    floss <- function(y1, y2) {
      ifelse(task == "classification", mean(y1 != y2), mean(abs(y1 - y2)))
    }
  }

  # Run (and compare) all methods:
  runOne <- function(o) {
    o(dataTrain, dataTest, targetTrain, targetTest, verbose, CV, floss, ...)
  }
  if (is.list(method_s))
    errors <- sapply(method_s, runOne)
  else if (is.function(method_s))
    errors <- runOne(method_s)
  else
    errors <- c()
  invisible(errors)
}

# Run compareTo N times in parallel
compareMulti <- function(
  data, target, N = 100, nc = NA,
  CV = list(type = "MC",
            V = 10,
            test_size = 0.2,
            shuffle = TRUE),
  method_s=NULL, ...
) {
  if (is.na(nc))
    nc <- parallel::detectCores()
  compareOne <- function(n) {
    print(n)
    compareTo(data, target, n, verbose=FALSE, CV, method_s, ...)
  }
  errors <- if (nc >= 2) {
    require(parallel)
    parallel::mclapply(1:N, compareOne, mc.cores = nc)
  } else {
    lapply(1:N, compareOne)
  }
  print("Errors:")
  Reduce('+', errors) / N
}

# TODO: unfinished !
