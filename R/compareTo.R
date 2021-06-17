standardCV_core <- function(data, target, task, gmodel, params, loss, CV) {
  n <- nrow(data)
  shuffle_inds <- NULL
  if (CV$type == "vfold" && CV$shuffle)
    shuffle_inds <- sample(n, n)
  list_testinds <- list()
  for (v in seq_len(CV$V))
    list_testinds[[v]] <- get_testIndices(n, CV, v, shuffle_inds)
  gmodel <- agghoo::Model$new(data, target, task, gmodel, params)
  best_error <- Inf
  best_model <- NULL
  for (p in seq_len(gmodel$nmodels)) {
    error <- Reduce('+', lapply(seq_len(CV$V), function(v) {
      testIdx <- list_testinds[[v]]
      d <- splitTrainTest(data, target, testIdx)
      model_pred <- gmodel$get(d$dataTrain, d$targetTrain, p)
      prediction <- model_pred(d$dataTest)
      loss(prediction, d$targetTest)
    }) )
    if (error <= best_error) {
      newModel <- list(model=gmodel$get(data, target, p),
                       param=gmodel$getParam(p))
      if (error == best_error)
        best_model[[length(best_model)+1]] <- newModel
      else {
        best_model <- list(newModel)
        best_error <- error
      }
    }
  }
#browser()
  best_model[[ sample(length(best_model), 1) ]]
}

standardCV_run <- function(
  dataTrain, dataTest, targetTrain, targetTest, CV, floss, verbose, ...
) {
  args <- list(...)
  task <- checkTask(args$task, targetTrain)
  modPar <- checkModPar(args$gmodel, args$params)
  loss <- checkLoss(args$loss, task)
  s <- standardCV_core(
    dataTrain, targetTrain, task, modPar$gmodel, modPar$params, loss, CV)
  if (verbose)
    print(paste( "Parameter:", s$param ))
  p <- s$model(dataTest)
  err <- floss(p, targetTest)
  if (verbose)
    print(paste("error CV:", err))
  invisible(err)
}

agghoo_run <- function(
  dataTrain, dataTest, targetTrain, targetTest, CV, floss, verbose, ...
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
  invisible(err)
}

# ... arguments passed to method_s (agghoo, standard CV or else)
compareTo <- function(
  data, target, method_s, rseed=-1, floss=NULL, verbose=TRUE, ...
) {
  if (rseed >= 0)
    set.seed(rseed)
  n <- nrow(data)
  test_indices <- sample( n, round(n / ifelse(n >= 500, 10, 5)) )
  d <- splitTrainTest(data, target, test_indices)
  CV <- checkCV(list(...)$CV)

  # Set error function to be used on model outputs (not in core method)
  task <- checkTask(list(...)$task, target)
  if (is.null(floss)) {
    floss <- function(y1, y2) {
      ifelse(task == "classification", mean(y1 != y2), mean(abs(y1 - y2)))
    }
  }

  # Run (and compare) all methods:
  runOne <- function(o) {
    o(d$dataTrain, d$dataTest, d$targetTrain, d$targetTest,
      CV, floss, verbose, ...)
  }
  errors <- c()
  if (is.list(method_s))
    errors <- sapply(method_s, runOne)
  else if (is.function(method_s))
    errors <- runOne(method_s)
  invisible(errors)
}

# Run compareTo N times in parallel
# ... : additional args to be passed to method_s
compareMulti <- function(
  data, target, method_s, N=100, nc=NA, floss=NULL, ...
) {
  require(parallel)
  if (is.na(nc))
    nc <- parallel::detectCores()

  # "One" comparison for each method in method_s (list)
  compareOne <- function(n) {
    print(n)
    compareTo(data, target, method_s, n, floss, verbose=FALSE, ...)
  }

  errors <- if (nc >= 2) {
    parallel::mclapply(1:N, compareOne, mc.cores = nc)
  } else {
    lapply(1:N, compareOne)
  }
  print("Errors:")
  Reduce('+', errors) / N
}
